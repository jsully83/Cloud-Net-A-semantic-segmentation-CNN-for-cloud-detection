from tensorflow import stack
from tensorflow.keras.losses import Loss
from tensorflow.math import log, reduce_sum

from keras import backend as K


smooth = 0.0000001
e = 2.7182818284590452353602874713527
m = 1000
p_c = 0.5

def jacc_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)

    return 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))

def fjl_inv_jacc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_comp = K.flatten((1-y_true))
    y_pred_comp = K.flatten((1-y_pred))
    
    S = K.sum(y_true_f)
    intersection = K.sum(y_true_f * y_pred_f)
    intersection_comp = K.sum(y_true_comp * y_pred_comp)
    
    inv_jac = 1 - ((intersection_comp + smooth) / (K.sum(y_true_comp) + K.sum(y_pred_comp) - intersection_comp + smooth))
    soft_jac = 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))

    HPF = 1/(1+pow(e, 1000*(-S+0.5)))
    LPF = 1/(1+pow(e, 1000*(S-0.5)))

    return (inv_jac * LPF) + (soft_jac * HPF)

def fjl_norm_ce(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    S = K.sum(y_true_f)
    N = len(y_true_f)
    intersection = K.sum(y_true_f * y_pred_f)
    max = K.cast(K.log(smooth), dtype='float32')

    norm_ce = (K.cast((-1/N), dtype='float32') * K.sum(K.cast(y_pred_f, dtype='float32') * K.cast(K.log(y_true_f + smooth), dtype='float32'))) / max
    soft_jac = 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))

    HPF = 1/(1+pow(e, 1000*(-S+0.5)))
    LPF = 1/(1+pow(e, 1000*(S-0.5)))

    return (norm_ce * LPF) + (soft_jac * HPF)

class AsymmetricUnifiedFocalLoss (Loss):
    """
    The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def __init__(self, weight, delta, gamma, name='asymmetric_unified_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.epsilon = K.epsilon()
        # print('init', self.weight, self.delta, self.gamma, self.epsilon)

    def __call__(self, y_true, y_pred, sample_weight=None):
        # print('call', y_true.get_shape(), y_pred.get_shape())
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        y_true_b = 1 - y_true_f
        y_pred_b = 1 - y_pred_f

        self.y_true_s = stack((y_true_f, y_true_b))
        self.y_pred_s = stack((y_pred_f, y_pred_b))
        self.y_pred_s = K.clip(self.y_pred_s, self.epsilon, 1. - self.epsilon)
    
        asymmetric_ftl = self.asymmetric_focal_tversky_loss()
        asymmetric_fl = self.asymmetric_focal_loss()

        if self.weight is not None:
            return (self.weight * asymmetric_ftl) + ((1-self.weight) * asymmetric_fl)
        else:
            return (asymmetric_ftl + asymmetric_fl)

    def asymmetric_focal_tversky_loss(self):
        """
        This is the implementation for binary segmentation.
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7
        gamma : float, optional
            focal parameter controls degree of down-weighting of easy examples, by default 0.75
        """

        tp = K.sum(self.y_true_s * self.y_pred_s, axis=1)
        fn = K.sum(self.y_true_s * (1-self.y_pred_s), axis=1)
        fp = K.sum((1-self.y_true_s) * self.y_pred_s, axis=1)

        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[1]) 
        fore_dice = (1-dice_class[0]) * K.pow(1-dice_class[1], -self.gamma) 

        # Average class scores
        loss = K.mean(stack([back_dice,fore_dice],axis=-1))
        # print('tversky loss: ', np.shape(loss), len(loss))
        return loss

    def asymmetric_focal_loss(self):
        """
        For Imbalanced datasets
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7
        gamma : float, optional
            Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
        """

        cross_entropy = -self.y_true_s * K.log(self.y_pred_s)

        #calculate losses separately for each class, only suppressing background class
        back_ce = K.pow(1 - self.y_pred_s[1,:], self.gamma) * cross_entropy[1,:]
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = cross_entropy[0,:]
        fore_ce = self.delta * fore_ce

        loss = K.mean(K.sum(stack([back_ce, fore_ce],axis=-1),axis=-1))

        return loss


class WeightedCrossEntropy(Loss):
    def __init__(self, weight1, weight2, name='weighed_cross_entropy', **kwargs):
        super().__init__(name=name, **kwargs)
        
        # find these weights before training
        self.weight1 = weight1
        self.weight2 = weight2

    def call(y_true, y_pred):
        class_weight1 = self.weight1 * y_true
        class_weight0 = self.weight2 * (1-y_true)

        term1 = -y_true * log(y_pred) * class_weight1 # clouds total pixels/
        term2 = (1-y_true) * log(1-y_pred) * class_weight2
        return reduce_sum(term1 - term2)


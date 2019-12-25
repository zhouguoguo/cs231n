from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #pass
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
        X_i =  X[i,:] # 1*D
        
        score_i = X_i.dot(W)  # 1*C
        stability = -score_i.max()
        exp_score_i = np.exp(score_i+stability)
        
        exp_score_total_i = np.sum(exp_score_i , axis = 0)
        
        for j in xrange(num_classes):
            p_j = exp_score_i[j] / exp_score_total_i
            if j == y[i]:
                dW[:, j] += -X_i.T + p_j * X_i.T  ##  正确类别的梯度是softmax值p_j-1
            else:
                dW[:, j] += p_j * X_i.T           ##  其他类别的梯度和softmax值相等
                
        prob = exp_score_i[y[i]] / exp_score_total_i
        loss += -np.log(prob)
        
        # dW[:, y[i]] += X[i,:] * (es[y[i]] - 1)
    
    loss /= float(num_train)
    loss += reg * np.sum(W * W)
    
    dW = dW / float(num_train) + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #dW = np.zeros(W.shape)
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #pass
    score = X.dot(W)  # N*D dot D*C  => N*C 每行是一个样本的score
    score += - np.max(score , axis=1)[..., None]  # 减去 每行最大值, (求每行最大值, axis=1)
    exp_score = np.exp(score) # N*C 每行是一个样本的exp_score
    sum_exp_score_col = np.sum(exp_score , axis = 1) # N*1 算出每个样本的exp_score总和, (计算每行的和, axis=1)
    
    loss = np.log(sum_exp_score_col) # log(sum) (N,)
    #print(loss.shape)
    #print(score.shape)
    loss = loss - score[np.arange(num_train), y] # 减去label的score  => L_i = -f_yi + log(sum_exp_score)
    loss = np.sum(loss) / float(num_train) + 0.5 * reg * np.sum(W*W)
    
    Grad = exp_score / sum_exp_score_col[..., None]  # 每行是一个样本算出的softmax值 p_i  N*C
    Grad[np.arange(num_train), y] += -1.0  # 每个样本的label类别p_i值减1
    dW = np.dot(X.T, Grad) / float(num_train) + reg*W    # D*N dot N*C => D*C 
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

#!usr/bin/env python


import itertools  # 迭代工具
import numpy as np  # 科学计算库
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  



class gcForest(object):
    # 定义初始化方法
    def __init__(self, shape_1X=None, n_mgsRFtree=30, window=None, stride=1,
                 cascade_test_size=0.2, n_cascadeRF=2, n_cascadeRFtree=101, cascade_layer=np.inf,
                 min_samples_mgs=0.1, min_samples_cascade=0.05, tolerance=0.0, n_jobs=1):
        """ gcForest Classifier.

        :param shape_1X: int or tuple list or np.array (default=None)
            Shape of a single sample element [n_lines, n_cols]. Required when calling mg_scanning!
            For sequence data a single int can be given.

        :param n_mgsRFtree: int (default=30)
            Number of trees in a Random Forest during Multi Grain Scanning.

        :param window: int (default=None)
            List of window sizes to use during Multi Grain Scanning.
            If 'None' no slicing will be done.

        :param stride: int (default=1)
            Step used when slicing the data.

        :param cascade_test_size: float or int (default=0.2)
            Split fraction or absolute number for cascade training set splitting.

        :param n_cascadeRF: int (default=2)
            Number of Random Forests in a cascade layer.
            For each pseudo Random Forest a complete Random Forest is created, hence
            the total numbe of Random Forests in a layer will be 2*n_cascadeRF.

        :param n_cascadeRFtree: int (default=101)
            Number of trees in a single Random Forest in a cascade layer.

        :param min_samples_mgs: float or int (default=0.1)
            Minimum number of samples in a node to perform a split
            during the training of Multi-Grain Scanning Random Forest.
            If int number_of_samples = int.
            If float, min_samples represents the fraction of the initial n_samples to consider.

        :param min_samples_cascade: float or int (default=0.1)
            Minimum number of samples in a node to perform a split
            during the training of Cascade Random Forest.
            If int number_of_samples = int.
            If float, min_samples represents the fraction of the initial n_samples to consider.

        :param cascade_layer: int (default=np.inf)
            mMximum number of cascade layers allowed.
            Useful to limit the contruction of the cascade.

        :param tolerance: float (default=0.0)
            Accuracy tolerance for the casacade growth.
            If the improvement in accuracy is not better than the tolerance the construction is
            stopped.

        :param n_jobs: int (default=1)
            The number of jobs to run in parallel for any Random Forest fit and predict.
            If -1, then the number of jobs is set to the number of cores.
        """
        setattr(self, 'shape_1X', shape_1X)  # 设置属性值
        setattr(self, 'n_layer', 0)  # 设置属性值
        setattr(self, '_n_samples', 0)  # 设置属性值
        setattr(self, 'n_cascadeRF', int(n_cascadeRF))  # 设置属性值
        if isinstance(window, int):  # 判断
            setattr(self, 'window', [window])  # 设置属性值
        elif isinstance(window, list):  # 判断
            setattr(self, 'window', window)  # 设置属性值
        setattr(self, 'stride', stride)  # 设置属性值
        setattr(self, 'cascade_test_size', cascade_test_size)  # 设置属性值
        setattr(self, 'n_mgsRFtree', int(n_mgsRFtree))  # 设置属性值
        setattr(self, 'n_cascadeRFtree', int(n_cascadeRFtree))  # 设置属性值
        setattr(self, 'cascade_layer', cascade_layer)  # 设置属性值
        setattr(self, 'min_samples_mgs', min_samples_mgs)  # 设置属性值
        setattr(self, 'min_samples_cascade', min_samples_cascade)  # 设置属性值
        setattr(self, 'tolerance', tolerance)  # 设置属性值
        setattr(self, 'n_jobs', n_jobs)  # 设置属性值

    # 定义拟合方法
    def fit(self, X, y):
        """ Training the gcForest on input data X and associated target y.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array
            1D array containing the target values.
            Must be of shape [n_samples]
        """
        if np.shape(X)[0] != len(y):  # 判断
            raise ValueError('Sizes of y and X do not match.')  # 抛出错误

        mgs_X = self.mg_scanning(X, y)  # 调用mg_scanning方法
        _ = self.cascade_forest(mgs_X, y)  # 调用cascade_forest方法

    # 定义预测方法
    def predict_proba(self, X):
        """ Predict the class probabilities of unknown samples X.

        :param X: np.array
            Array containing the input samples.
            Must be of the same shape [n_samples, data] as the training inputs.

        :return: np.array
            1D array containing the predicted class probabilities for each input sample.
        """
        mgs_X = self.mg_scanning(X)  # 调用mg_scanning方法
        cascade_all_pred_prob = self.cascade_forest(mgs_X)  # 调用cascade_forest方法
        predict_proba = np.mean(cascade_all_pred_prob, axis=0)  # 赋值

        return predict_proba  # 返回数据

    # 定义预测方法
    def predict(self, X):
        """ Predict the class of unknown samples X.

        :param X: np.array
            Array containing the input samples.
            Must be of the same shape [n_samples, data] as the training inputs.

        :return: np.array
            1D array containing the predicted class for each input sample.
        """
        pred_proba = self.predict_proba(X=X)  # 调用predict_proba方法
        predictions = np.argmax(pred_proba, axis=1)  # 获取预测值

        return predictions  # 返回预测数

    # 定义 对输入数据执行多粒度扫描 方法
    def mg_scanning(self, X, y=None):
        """ Performs a Multi Grain Scanning on input data.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)

        :return: np.array
            Array of shape [n_samples, .. ] containing Multi Grain Scanning sliced data.
        """
        setattr(self, '_n_samples', np.shape(X)[0])  # 设置属性值
        shape_1X = getattr(self, 'shape_1X')  # 获取属性值
        if isinstance(shape_1X, int):  # 判断
            shape_1X = [1, shape_1X]  # 赋值
        if not getattr(self, 'window'):  # 判断
            setattr(self, 'window', [shape_1X[1]])  # 设置属性值

        mgs_pred_prob = []  # 定义一个空列表

        for wdw_size in getattr(self, 'window'):  # 判断
            wdw_pred_prob = self.window_slicing_pred_prob(X, wdw_size, shape_1X, y=y)  # 调用window_slicing_pred_prob方法
            mgs_pred_prob.append(wdw_pred_prob)  # 数据追加到列表

        return np.concatenate(mgs_pred_prob, axis=1)  # 返回数据

    
    def window_slicing_pred_prob(self, X, window, shape_1X, y=None):
        """ Performs a window slicing of the input data and send them through Random Forests.
        If target values 'y' are provided sliced data are then used to train the Random Forests.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample.

        :param y: np.array (default=None)
            Target values. If 'None' no training is done.

        :return: np.array
            Array of size [n_samples, ..] containing the Random Forest.
            prediction probability for each input sample.
        """
        n_tree = getattr(self, 'n_mgsRFtree')  # 获取属性值
        min_samples = getattr(self, 'min_samples_mgs')  # 获取属性值
        stride = getattr(self, 'stride')  # 获取属性值

        if shape_1X[0] > 1:  # 判断
            print('Slicing Images...')  # 输出信息
            sliced_X, sliced_y = self._window_slicing_img(X, window, shape_1X, y=y,
                                                          stride=stride)  # 调用_window_slicing_img方法
        else:  # 判断
            print('Slicing Sequence...')  # 输出信息
            sliced_X, sliced_y = self._window_slicing_sequence(X, window, shape_1X, y=y,
                                                               stride=stride)  # 调用_window_slicing_sequence

        if y is not None:  # 判断
            n_jobs = getattr(self, 'n_jobs')  # 获取属性值
            prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                         min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)  # 建模
            crf = RandomForestClassifier(n_estimators=n_tree, max_features=1,
                                         min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)  # 建模
            print('Training MGS Random Forests...')  # 输出信息
            prf.fit(sliced_X, sliced_y)  # 拟合
            crf.fit(sliced_X, sliced_y)  # 拟合
            setattr(self, '_mgsprf_{}'.format(window), prf)  # 设置属性值
            setattr(self, '_mgscrf_{}'.format(window), crf)  # 设置属性值
            pred_prob_prf = prf.oob_decision_function_  # 预测值
            pred_prob_crf = crf.oob_decision_function_  # 预测值

        if hasattr(self, '_mgsprf_{}'.format(window)) and y is None:  # 判断
            prf = getattr(self, '_mgsprf_{}'.format(window))  # 获取属性值
            crf = getattr(self, '_mgscrf_{}'.format(window))  # 获取属性值
            pred_prob_prf = prf.predict_proba(sliced_X)  # 进行预测
            pred_prob_crf = crf.predict_proba(sliced_X)  # 进行预测

        pred_prob = np.c_[pred_prob_prf, pred_prob_crf]  # 创建数组

        return pred_prob.reshape([getattr(self, '_n_samples'), -1])  # 返回数据

    # 定义数据切片处理方法
    def _window_slicing_img(self, X, window, shape_1X, y=None, stride=1):
        """ Slicing procedure for images

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample [n_lines, n_cols].

        :param y: np.array (default=None)
            Target values.

        :param stride: int (default=1)
            Step used when slicing the data.

        :return: np.array and np.array
            Arrays containing the sliced images and target values (empty if 'y' is None).
        """
        if any(s < window for s in shape_1X):  # 判断
            raise ValueError('window must be smaller than both dimensions for an image')  # 抛出数据

        len_iter_x = np.floor_divide((shape_1X[1] - window), stride) + 1  # 计算赋值
        len_iter_y = np.floor_divide((shape_1X[0] - window), stride) + 1  # 计算赋值
        iterx_array = np.arange(0, stride * len_iter_x, stride)  # 生成数据赋值
        itery_array = np.arange(0, stride * len_iter_y, stride)  # 生成数据赋值

        ref_row = np.arange(0, window)  # 生成数据赋值
        ref_ind = np.ravel([ref_row + shape_1X[1] * i for i in range(window)])  # 转为一维数据赋值
        inds_to_take = [ref_ind + ix + shape_1X[1] * iy
                        for ix, iy in itertools.product(iterx_array, itery_array)]  # 计算赋值

        sliced_imgs = np.take(X, inds_to_take, axis=1).reshape(-1, window ** 2)  # 从一维数组或更高维度数组中按照指定的索引抽取元素

        if y is not None:  # 判断
            sliced_target = np.repeat(y, len_iter_x * len_iter_y)  # 数组中的元素按照指定次数进行重复
        elif y is None:  # 判断
            sliced_target = None  # 赋值

        return sliced_imgs, sliced_target  # 返回数据

    # 定义序列的切片操作方法
    def _window_slicing_sequence(self, X, window, shape_1X, y=None, stride=1):
        """ Slicing procedure for sequences (aka shape_1X = [.., 1]).

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample [n_lines, n_col].

        :param y: np.array (default=None)
            Target values.

        :param stride: int (default=1)
            Step used when slicing the data.

        :return: np.array and np.array
            Arrays containing the sliced sequences and target values (empty if 'y' is None).
        """
        if shape_1X[1] < window:  # 判断
            raise ValueError('window must be smaller than the sequence dimension')  # 抛出错误

        len_iter = np.floor_divide((shape_1X[1] - window), stride) + 1  # 计算赋值
        iter_array = np.arange(0, stride * len_iter, stride)  # 生成数据并赋值

        ind_1X = np.arange(np.prod(shape_1X))  # 生成数据并赋值
        inds_to_take = [ind_1X[i:i + window] for i in iter_array]  # 循环并赋值
        sliced_sqce = np.take(X, inds_to_take, axis=1).reshape(-1, window)  # 从一维数组或更高维度数组中按照指定的索引抽取元素

        if y is not None:  # 判断
            sliced_target = np.repeat(y, len_iter)  # 数组中的元素按照指定次数进行重复
        elif y is None:  # 判断
            sliced_target = None  # 赋值

        return sliced_sqce, sliced_target  # 返回数据

    # 定义级联随机森林方法
    def cascade_forest(self, X, y=None):
        """ Perform (or train if 'y' is not None) a cascade forest estimator.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)
            Target values. If 'None' perform training.

        :return: np.array
            1D array containing the predicted class for each input sample.
        """
        if y is not None:  # 判断
            setattr(self, 'n_layer', 0)  # 设置属性赋值
            test_size = getattr(self, 'cascade_test_size')  # 获取属性值
            max_layers = getattr(self, 'cascade_layer')  # 获取属性值
            tol = getattr(self, 'tolerance')  # 获取属性值

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)  # 数据集拆分工具

            self.n_layer += 1  # 赋值
            prf_crf_pred_ref = self._cascade_layer(X_train, y_train)  # 调用_cascade_layer方法
            accuracy_ref = self._cascade_evaluation(X_test, y_test)  # 调用_cascade_evaluation方法
            feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)  # 调用_create_feat_arr

            self.n_layer += 1  # 赋值
            prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)  # 调用_cascade_layer方法
            accuracy_layer = self._cascade_evaluation(X_test, y_test)  # 调用_cascade_evaluation方法

            while accuracy_layer > (accuracy_ref + tol) and self.n_layer <= max_layers:  # 循环
                accuracy_ref = accuracy_layer  # 赋值
                prf_crf_pred_ref = prf_crf_pred_layer  # 赋值
                feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)  # 调用_create_feat_arr方法
                self.n_layer += 1  # 赋值
                prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)  # 调用_cascade_layer方法
                accuracy_layer = self._cascade_evaluation(X_test, y_test)  # 调用_cascade_evaluation方法

            if accuracy_layer < accuracy_ref:  # 判断
                n_cascadeRF = getattr(self, 'n_cascadeRF')  # 获取属性值
                for irf in range(n_cascadeRF):  # 循环
                    delattr(self, '_casprf{}_{}'.format(self.n_layer, irf))  # 删除对象属性
                    delattr(self, '_cascrf{}_{}'.format(self.n_layer, irf))  # 删除对象属性
                self.n_layer -= 1  # 赋值

        elif y is None:  # 判断
            at_layer = 1  # 赋值
            prf_crf_pred_ref = self._cascade_layer(X, layer=at_layer)  # 调用_cascade_layer方法
            while at_layer < getattr(self, 'n_layer'):  # 循环
                at_layer += 1  # 赋值
                feat_arr = self._create_feat_arr(X, prf_crf_pred_ref)  # 调用_create_feat_arr方法
                prf_crf_pred_ref = self._cascade_layer(feat_arr, layer=at_layer)  # 调用_cascade_layer方法

        return prf_crf_pred_ref  # 返回数据

    # 定义包含随机森林estimators的级联层方法
    def _cascade_layer(self, X, y=None, layer=0):
        """ Cascade layer containing Random Forest estimators.
        If y is not None the layer is trained.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)
            Target values. If 'None' perform training.

        :param layer: int (default=0)
            Layer indice. Used to call the previously trained layer.

        :return: list
            List containing the prediction probabilities for all samples.
        """
        n_tree = getattr(self, 'n_cascadeRFtree')  # 获取属性值
        n_cascadeRF = getattr(self, 'n_cascadeRF')  # 获取属性值
        min_samples = getattr(self, 'min_samples_cascade')  # 获取属性值

        n_jobs = getattr(self, 'n_jobs')  # 获取属性值
        prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                     min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)  # 建模
        crf = RandomForestClassifier(n_estimators=n_tree, max_features=1,
                                     min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)  # 建模

        prf_crf_pred = []  # 定义空列表
        if y is not None:  # 判断
            print('Adding/Training Layer, n_layer={}'.format(self.n_layer))  # 输出信息
            for irf in range(n_cascadeRF):  # 循环
                prf.fit(X, y)  # 拟合
                crf.fit(X, y)  # 拟合
                setattr(self, '_casprf{}_{}'.format(self.n_layer, irf), prf)  # 设置属性并赋值
                setattr(self, '_cascrf{}_{}'.format(self.n_layer, irf), crf)  # 设置属性并赋值
                prf_crf_pred.append(prf.oob_decision_function_)  # 添加预测值到列表
                prf_crf_pred.append(crf.oob_decision_function_)  # 添加预测值到列表
        elif y is None:  # 判断
            for irf in range(n_cascadeRF):  # 循环
                prf = getattr(self, '_casprf{}_{}'.format(layer, irf))  # 获取属性值
                crf = getattr(self, '_cascrf{}_{}'.format(layer, irf))  # 获取属性值
                prf_crf_pred.append(prf.predict_proba(X))  # 添加预测值到列表
                prf_crf_pred.append(crf.predict_proba(X))  # 添加预测值到列表

        return prf_crf_pred  # 返回数据

    # 定义级联模型评估方法
    def _cascade_evaluation(self, X_test, y_test):
        """ Evaluate the accuracy of the cascade using X and y.

        :param X_test: np.array
            Array containing the test input samples.
            Must be of the same shape as training data.

        :param y_test: np.array
            Test target values.

        :return: float
            the cascade accuracy.
        """
        casc_pred_prob = np.mean(self.cascade_forest(X_test), axis=0)  # 计算并赋值
        casc_pred = np.argmax(casc_pred_prob, axis=1)  # 获取预测值
        casc_accuracy = accuracy_score(y_true=y_test, y_pred=casc_pred)  # 计算准确率
        print('Layer validation accuracy = {}'.format(casc_accuracy))  # 输出模型评估指标数值

        return casc_accuracy  # 返回数据

    # 定义将原始特征向量与级联层的预测值连接起来
    def _create_feat_arr(self, X, prf_crf_pred):
        """ Concatenate the original feature vector with the predicition probabilities
        of a cascade layer.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param prf_crf_pred: list
            Prediction probabilities by a cascade layer for X.

        :return: np.array
            Concatenation of X and the predicted probabilities.
            To be used for the next layer in a cascade forest.
        """
        swap_pred = np.swapaxes(prf_crf_pred, 0, 1)  # 计算并赋值
        add_feat = swap_pred.reshape([np.shape(X)[0], -1])  # 改变数组维度
        feat_arr = np.concatenate([add_feat, X], axis=1)  # 连接数组

        return feat_arr  # 返回数据

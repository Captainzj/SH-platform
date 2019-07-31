from MeDas.Alg.common.test import *
from MeDas.Alg.preprocessing.ArrayResize import *
from MeDas.Alg.postprocessing.BinarizeLabel import *
from MeDas.Alg.preprocessing.BinaryNormalization import *
from MeDas.Alg.debugging.CalcRANO import *
from MeDas.Alg.debugging.CalcVolume import *
from MeDas.Alg.debugging.CalcVoxelCount import *
from src.MeDas.Alg.datamanagement.ChooseData_Number import *
from MeDas.Alg.visualization.CreateMosaic import *
from MeDas.Alg.datamanagement.DatasetSplit import *
from MeDas.Alg.postprocessing.ErrorCalculation import *
from MeDas.Alg.postprocessing.FillHoles import *
from MeDas.Alg.datamanagement.GeoTransMirror import *
from MeDas.Alg.augmentation.Grab_Patchs import *
from MeDas.Alg.visualization.LabelEdge import *
from MeDas.Alg.postprocessing.LargestComponents import *
# from MeDas.Alg.debugging.LiverSegmentation import *   # ...
from MeDas.Alg.datamanagement.Masking import *
from MeDas.Alg.preprocessing.ZeroMeanNormalization import *
from MeDas.Alg.preprocessing.MergeChannels import *
from MeDas.Alg.preprocessing.Normalization import *
from MeDas.Alg.preprocessing.PetCtRegistration import PetCtRegistration
from MeDas.Alg.augmentation.RandomGaussian import RandomGaussian
from MeDas.Alg.preprocessing.Resample import *
from MeDas.Alg.postprocessing.Rescale import Rescale
from MeDas.Alg.segmentation.RegionGrow import *
from MeDas.Alg.segmentation.RandomWalk.PET import *
from MeDas.Alg.preprocessing.SelectChannels import *
from MeDas.Alg.application.SliceFitting import *
from MeDas.Alg.preprocessing.SplitData import *
from MeDas.Alg.preprocessing.ThresholdProcessing import *


import numpy as np


class test(MeDasTestCase):

    def succ(self, out):
        print("UnitTest_AD_nii OUT", out)
    def fail(self, error):
        self.fail('catch error %s' % error)

    # def test_ArrayResize(self):
    #     """
    #     @brief 重新调整矩阵，将矩阵进行缩放
    #     @brief order的参数为0-5之间：[0,5]
    #     0：最近邻插值
    #     1：样条插值算法
    #     2; 二次样条插值
    #     3：三次（立方）插值
    #     4：四次样条插值
    #     5：五次样条插值
    #     @param zoom_0 第一个维度上进行缩放的比例，默认值为 1
    #     @param zoom_1 第二个维度上进行缩放的比例，默认值为 1
    #     @param zoom_2 第三个维度上进行缩放的比例，默认值为 1
    #     @param zoom_3 第四个维度上进行缩放的比例，默认值为 1
    #     @param
    #     """
    #     order = -5
    #     zoom_0. zoom_1, zoom_2 = 1.0, 2.0, 3.0
    #
    #     result = ArrayResize().set_params(vec = '/Users/Captain/Desktop/ADNI_test_1.nii')\
    #         .set_params(order = order, zoom_0 = zoom_0, zoom_1 = zoom_1, zoom_2 = zoom_2).run() \
    #         .get_result(file='ADNI_test_1_ArrayResize', itype='nii', dtype=np.int16) \
    #         .with_succ(self.succ).with_fail(self.fail)

    def test_BinarizeLabel(self):
        """
        @brief 对输入的矩阵进行标签二值化
    
        @param input_data                 输入的待处理矩阵， ndarray类型，默认值为    None
        @param binarization_threshold     输入的阈值，      浮点数，      默认值为   0.5
    
        """
        binarization_threshold = 255.0
        result = BinarizeLabel().set_params(input_data='/Users/Captain/Desktop/ADNI_test_1.nii') \
            .set_params(binarization_threshold = binarization_threshold).run() \
            .get_result(file='ADNI_test_1_BinarizeLabel', itype='nii', dtype=np.int16) \
            .with_succ(self.succ).with_fail(self.fail)


    # def test_BinaryNormalization(self):
    #     """
    #     @brief 对输入的矩阵进行二值归一化，归一化的模式有两种，一种是 根据阈值，将大于阈值的元素设置为元素上限，将小于等于阈值的元素设置为元素下限。
    #            第二种是根据值x，将等于阈值的元素设置为元素下限，将不等于的元素设置为元素下限。（元素上限不一定大于元素下限）
    #
    #     @param input_data                 输入的待处理矩阵， ndarray类型，默认值为    None
    #     @param threshold_value            输入的阈值，      浮点数，      默认值为   0.5
    #     @param up_bound                   输入的元素上限，      浮点数，      默认值为   1
    #     @param down_bound                 输入的元素下限，      浮点数，      默认值为   0
    #     @param model                      输入的选择模式，      bool型，      默认值为   True
    #
    #     """
    #     threshold_value = 300
    #     up_bound = 600
    #     down_bound = 1
    #     model = True
    #     result = BinaryNormalization().set_params(input_data='/Users/Captain/Desktop/ADNI_test_1.nii') \
    #         .set_params(threshold_value = threshold_value, up_bound = up_bound, down_bound = down_bound, model = model).run() \
    #         .get_result(file='ADNI_test_1_BinaryNormalization', itype='nii', dtype=np.int16) \
    #         .with_succ(self.succ).with_fail(self.fail)

    #
    # # def test_CalcRANO(self):  #...
    # #     """
    # #     @brief   计算RANO（神经肿瘤反应评价）值，RANO是2010年哈佛医学院的Wen Patrick Y教授牵头的神经肿瘤治疗反应评估工作组提出的
    # #              新的高级别胶质肿瘤反应评价标准，即RANO标准。
    # #              RANO标准是以二维评价标准进行治疗反应的评估，对于增强病灶（本方法主要用于计算增强病灶评估计算），采用最大横截面下两垂直
    # #              直径的乘积来界定肿瘤的大小，多病灶取乘积之和。
    # #              可测量强化病灶被界定为CT或MRI上边界明确的增强病灶，能够在层厚为5 mm的≥2张轴位片上显影，且相互垂直的长径均>10 mm。
    # #              如扫描层厚较大，最小可测量病灶应>2倍层厚。
    # #              关于病灶数量：如患者存在多个增强病灶，应至少测量2个最大的病灶，然后将各自最大截面下垂直直径的乘积相加。鉴于一些病灶
    # #              难以测量以及高级别胶质瘤的异质性，最多仅对其中最大的5个病灶进行测量，且应包含最大的增强病灶。（在本方法中，最多选取5个）
    # #
    # #
    # #     @param input_data    输入数据                ndarray  默认值 None
    # #     @param resolution_x  像素间在x轴上的单位距离   float    默认值：1
    # #     @param resolution_y  像素间在y轴上的单位距离   float    默认值：1
    # #     @param resolution_z  像素间在z轴上的单位距离   float    默认值：1
    # #     @param xyz           输入矩阵各轴对应关系      int      默认值：0      对于传入矩阵，一般来说由DICOM文件直接生成的矩阵，
    # #                                                                        轴排序为zxy，但有时也可能是xyz。在进行运算的时候，
    # #                                                                        将统一按照xyz进行运算，当xyz=0时，表示传入矩阵的
    # #                                                                        轴排序为zxy，需要进行调整，否则无需调整。
    # #     """
    # #     result = CalcRANO().set_params(input_data='/Users/Captain/Desktop/ADNI_test_1.nii').run() \
    # #             .with_succ(self.succ).with_fail(self.fail)
    #
    #
    # def test_CalcVolume(self):
    #     """
    #     @brief 计算体积
    #
    #     @param label 被统计体积的对象的 label
    #
    #     @param x_spacing X 轴向间隙（直接使用影像中的）
    #     @param y_spacing Y 轴向间隙（直接使用影像中的）
    #     @param z_spacing Z 轴向间隙（直接使用影像中的）
    #
    #     @param vec 输入影像，需要时三维空间的。
    #     """
    #     result = CalcVolume().set_params(vec='/Users/Captain/Desktop/ADNI_test_1.nii').run() \
    #         .get_result() \
    #         .with_succ(self.succ).with_fail(self.fail)   # 7246906.3701171875
    #
    #
    # def test_CalcVoxelCount(self):
    #     """
    #     @brief 计算RIO区域像素的个数：像素值不为0 或者 像素的label == label
    #     @param vec 输入影像或者某一张切片，需要是3D数据或者2D数据。
    #     """
    #     result = CalcVoxelCount().set_params(vec='/Users/Captain/Desktop/ADNI_test_1.nii').run()\
    #         .get_result()\
    #         .with_succ(self.succ).with_fail(self.fail)  # 6039104
    #
    #
    # def test_ChooseData_Number(self): # ✔️
    #     """
    #     @brief 选择一例数据中的部分切片用作训练(不需要使用全部切片)
    #
    #     @parama axis   输入在哪一个轴上进行选取，非负整数，默认值为0
    #     @parama number 输入选取多少张切片，非负整数，默认值为40
    #     @brief   vec: 待处理的矩阵（3D）
    #     """
    #     number = 100
    #
    #     result = ChooseData_Number().set_params(vec='/Users/Captain/Desktop/ADNI_test_1.nii')\
    #         .set_params(number=number).run()\
    #         .get_result(file='ADNI_test_1_ChooseData_Number', itype='nii', dtype=np.int16)\
    #         .with_succ(self.succ).with_fail(self.fail)
    #
    #
    # def test_CreateMosaic(self):  # ✔️
    #     """
    #     @brief 将3D图像数据拼接处理为2D图像数据
    #
    #     @param input_data    输入数据，  Numpy 矩阵或者是影像    一般为3D图像数据，即待处理的图像数据
    #     @param step          步长       int                  将三维数据按某一轴进行切片，每（步长）个切片选取一个，拼接成新的2D图像
    #     @param rotate_90     旋转角度    int                  将拼接的2D图像进行相应旋转，以90°为一个单位。一般默认的旋转角度为3个单位
    #     @param cols          图像列长度   int                  拼接完成后2D图像的列长度（每个长度单位表示原3D切片的列长度），
    #                                                           默认值为8个单位长度，也就是原来3D图像切片的列长度的8倍
    #     @param dim           选择的轴     int                  当3D图像拼接成2D图像时，会沿着这个维度进行切片，然后将切片进行拼接。
    #                                                           默认值为2
    #     @param flip          是否翻转     bool                  如果原始图像需要进行翻转，则对其进行相应翻转。默认值为TRUE
    #     """
    #     result = CreateMosaic().set_params(input_data='/Users/Captain/Desktop/ADNI_test_1.nii') \
    #             .set_params(step=1, rotate_90=3, cols=6, dim=0, flip=False).run()\  #dim=2
    #             .get_result(file='ADNI_test_1_CreateMosaic', itype='png', dtype=np.int16)\
    #             .with_succ(self.succ).with_fail(self.fail)
    #
    #
    # def test_DatasetSplit(self):  # 当count_Dataset=7时，proportion = (2, 1, 1)，输出比例为 3:1:3 与预期不符
    #     """
    #     @brief 划分数据集，返回三个集合的绝对路径
    #     @param proportion用户自定义的 训练集：验证集：测试集的比例，默认比例为6：2：2（顺序是固定的，不可以颠倒）
    #     @param DatasetPath 数据集的路径：该目录下应该是一个个单独的病人数据（默认是一个个以P开头的文件夹，文件夹下是DICOM文件）
    #     使用 tuple 类型存储：tuple中的元素不可修改（此工具中的大部分元素值都不可修改，如：输入的比例，文件路径），使用完后删除元组
    #     """
    #     proportion = (2, 1, 1)
    #     DatasetPath = "/Users/Captain/Desktop/MASKS_DICOM"
    #     result = DatasetSplit().set_params(proportion=proportion, DatasetPath=DatasetPath).run()\
    #             .get_result()\
    #             .with_succ(self.succ).with_fail(self.fail)

    #
    # # def test_ErrorCalculation(self):  # so slow  # have no result
    # #     """
    # #     @brief 此接口主要计算待处理矩阵(input_data)与真值矩阵(ground_truth)之间的误差。共有三种误差形式分别是：
    # #            1、相似度误差（dice_cost）：该误差主要度量两个矩阵的相似性，计算结果为2倍的交并比，
    # #               公式为：dice_cost=2*logical_and(input_data,ground_truth).sum()/(input_data.sum()+ground_truth.sum())
    # #               其中，logical_and表示逻辑与，.sum()表示矩阵中真值（非0值）的个数。
    # #               对于相似度误差，结果范围：0-1；当结果为1时误差最小，为0时误差最大
    # #            2、准确度误差（accuracy_cost）：该误差度量两个矩阵相同元个数，
    # #               公式为：accuracy_cost=np.sum(input_data==ground_truth)
    # #               .sum()表示矩阵中真值（非0值）的个数。
    # #               对于准确度误差，结果>=0,即结果越大越好(最大为矩阵中元素数量，且此时两个矩阵完全相同)。
    # #            3、聚类相似性误差（cluster_accuracy）：该误差主要度量两个矩阵中连通区域（即聚类）之间的相似性，将矩阵按连通区域划分，比较相同
    # #               的连通区域的数量比。
    # #               对于聚类相似性误差，结果为0-1，当结果为1的时候表示两个矩阵的连通区域相同。
    # #
    # #     @param input_data                 输入的待处理矩阵， ndarray类型，      默认值为    None
    # #     @param ground_truth               输入的真值，      ndarray类型，      默认值为    None
    # #
    # #     """
    # #     result = ErrorCalculation().set_params(input_data='/Users/Captain/Desktop/ADNI_test_1.nii') \
    # #             .set_params(ground_truth = '/Users/Captain/Desktop/ADNI_test_1.nii').run()\
    # #             .get_result()\
    # #             .with_succ(self.succ).with_fail(self.fail)
    #
    #
    # def test_FillHoles(self):
    #     """
    #        @brief 孔洞填充  核心函数处理，计算最大的连通区域
    #
    #        @param input_data   输入的待处理矩阵， ndarray类型，默认值为    None
    #
    #     """
    #     result = FillHoles().set_params(input_data='/Users/Captain/Desktop/ADNI_test_1.nii').run()\
    #             .get_result(file='ADNI_test_1_FillHoles', itype='nii', dtype=np.int16)\
    #             .with_succ(self.succ).with_fail(self.fail)
    #
    #
    # def test_GeoTransMirror(self):
    #     """
    #     @brief 将 tensor 按照特定轴进行镜像或者是翻转。
    #
    #     @param mirror_axes 镜像操作轴序号，list，默认值为 None 进行随机翻转
    #
    #     @param tensor      带操作 tensor
    #     """
    #     mirror_axes = [0]
    #     result = GeoTransMirror().set_params(tensor='/Users/Captain/Desktop/ADNI_test_1.nii')\
    #         .set_params(mirror_axes = mirror_axes).run() \
    #         .get_result(file='ADNI_test_1_GeoTransMirror', itype='nii', dtype=np.int16) \
    #         .with_succ(self.succ).with_fail(self.fail)
    #
    #
    # def test_Grab_Patchs(self):
    #     """
    #     @ brief 补丁提取：对一个原始的Array(2D 或者 3D)进行裁剪。数据增强的一种方式。
    #     @ param corner_tuple ：矩阵的一个角的坐标，裁剪的方向都默认为 右下(3D 和 2D), 元组类型, 默认值为 None
    #     @ param patch_shape  ：裁剪后的目标矩阵大小, 元组类型， 默认值为 None
    #     @ param vec          ：输入的图像矩阵
    #     """
    #     corner_coordinate = tuple([1,1,10])  # 裁剪的起始原点
    #     patch_shape = tuple([200,200,120])  # 猜测：origin (170,256,256)→ vec.shape(256,256,170)
    #     result = Grab_Patchs().set_params(vec='/Users/Captain/Desktop/ADNI_test_1.nii')\
    #         .set_params(corner_coordinate =  corner_coordinate, patch_shape = patch_shape).run() \
    #         .get_result(file='ADNI_test_1_Grab_Patchs', itype='nii', dtype=np.int16) \
    #         .with_succ(self.succ).with_fail(self.fail)

    #
    # # def test_LabelEdge(self):   # seg data 不合适
    # #     """
    # #     @brief 将分割结果处理得到边缘。
    # #
    # #     @param seg    分割结果，    Numpy 矩阵或者是影像
    # #     @param label  目标 label， int， 待获取边缘的 label， 默认为 None， 求解所有边缘
    # #     """
    # #     result = LabelEdge().set_params(seg='/Users/Captain/Desktop/ADNI_test_1.nii').run() \
    # #         .get_result(file='ADNI_test_1_LabelEdge', itype='nii', dtype=np.int16) \
    # #         .with_succ(self.succ).with_fail(self.fail)
    #
    #
    # def test_LargestComponents(self):   # so slow  # have no result
    #     """
    #        @brief 计算最大的连通区域
    #
    #        @param input_data                 输入的待处理矩阵， ndarray类型，默认值为    None
    #        @param binarization_threshold     输入的阈值，      浮点数，      默认值为   0.5
    #
    #     """
    #     result = LargestComponents().set_params(input_data='/Users/Captain/Desktop/ADNI_test_1.nii').run() \
    #         .get_result(file='ADNI_test_1_LargestComponents', itype='nii', dtype=np.int16) \
    #         .with_succ(self.succ).with_fail(self.fail)

    #
    # # def test_LiverSegmentation(self):  # 待进一步的测试 # 报错！！！
    # #     """
    # #     @ brief   输入一个病人的CT数据，根据数据中的array信息和层厚z_spacing信息，得到肝轮廓的分割结果
    # #     """
    # #     result = LiverSegmentation().set_params(vec='/Users/Captain/Desktop/ADNI_test_1.nii').run() \
    # #         .get_result(file='ADNI_test_1_LiverSegmentation', itype='nii', dtype=np.int16) \
    # #         .with_succ(self.succ).with_fail(self.fail)
    #
    #
    # # def test_Masking(self):  # mask data 不合适
    # #     """
    # #     @brief 引用掩模至图像
    # #
    # #     @input  输入为多维数据、掩模，掩模的维数需要小于等于数据，当掩模数据维度与数据不相等时，需要提供维度信息。
    # #     @output 被掩模的数据。
    # #
    # #     @param tensor 输入数据 (np.ndarray)
    # #     @param mask   掩模     (np.ndarray)
    # #     @param axes   轴向数据 (dict)
    # #
    # #     @note 轴向数据是当掩模数据维度与数据不相等时使用，其他时候忽略。轴向数据的类型为字典，字典中键值分别是轴和相对设置。
    # #         轴为 numpy 中的轴，设置为选择通道，有三种方式：随机、指定、全部。如果键值为 None，则为全部，如果为 list，则是指定，
    # #         其他数据格式则是随机，（建议用 Bool 类型）。如果 mask 的维度和轴信息加起来，还有空缺（例如数据是4维的，mask 是1维的，
    # #         但是轴信息只定义了 1一个维度的，那么还有两个维度是没有定义的），则会报错：轴空缺。
    # #     """
    # #     result = Masking().set_params(tensor='/Users/Captain/Desktop/ADNI_test_1.nii')\
    # #         .set_params(mask='/Users/Captain/Desktop/ADNI_test_1.nii').run() \
    # #         .get_result(file='ADNI_test_1_Masking', itype='nii', dtype=np.int16) \
    # #         .with_succ(self.succ).with_fail(self.fail)


    # def test_ZeroMeanNormalization(self):   # [0,1]
    #     """
    #     @breif Zero Mean Normalization
    #         将数据整体减去均值，然后除以标准差。得到方差为 1，均值为 0 的数据的分布。
    #         $$\frac{x - \mu}{\sigma}$$
    #
    #     @param array 输入的数据
    #     @return 零均值话的数据
    #     @example
    #     ```python
    #         array = np.array([1,2,3,4,5])
    #         result = ZeroMeanNormalization().set_params(array = array).run().get_result()
    #     ```
    #     result.either 应该是 array([-1. , -0.5,  0. ,  0.5,  1. ])
    #     """
    #
    #     result = ZeroMeanNormalization().set_params(array='/Users/Captain/Desktop/ADNI_test_1.nii').run() \
    #         .get_result().with_succ(self.succ).with_fail(self.fail)


    # # MedicalFormatConvert...
    #
    #
    # def test_MergeChannels(self):  # 仅对第一个维度进行合并
    #     """
    #     @brief 将两个或两个以上的通道的数据进行合并，合并的形式包括max/min/sum/average四种。最终输出通道合并完成的矩阵。
    #     @param input_data                     输入的待处理矩阵，           ndarray类型，    默认值为    None
    #     @param channels                       待合并的通道，              list，          默认值为    None
    #     @param model                          合并的模式，                str字符串，      默认值为   max
    #     @param output_data                    输出参数（通道合并后的矩阵）  ndarray类型
    #     """
    #     channels = list(range(1,170,2))
    #     model = "average"
    #     result = MergeChannels().set_params(input_data='/Users/Captain/Desktop/ADNI_test_1.nii')\
    #         .set_params(channels=channels, model=model).run()\
    #         .get_result(file='ADNI_test_1_MergeChannels', itype='nii', dtype=np.int16) \
    #         .with_succ(self.succ).with_fail(self.fail)

    #
    # def test_Normalization(self):  # [0,1]
    #     """
    #     @brief 归一化，把数据变成（０，１）或者（-1,1）或者其他自定义之间的小数。主要是为了数据处理方便提出来的，
    #            把数据映射到0～1范围之内处理，更加便捷快速。
    #
    #     @param input_data                 输入的待处理矩阵，      ndarray类型，    默认值为    None
    #     @param intensity_range            输入的归一化范围，      list，          默认值为    [-1,1]
    #
    #     """
    #     intensity_range = [0., 1.]
    #     result = Normalization().set_params(input_data='/Users/Captain/Desktop/ADNI_test_1.nii')\
    #         .set_params(intensity_range=intensity_range).run()\
    #         .get_result() \
    #         .with_succ(self.succ).with_fail(self.fail)
    #
    # # PetCtRegistrationTest
    # # """
    # # @breif PET CT 图像的对齐
    # #
    # # @param ct 是 输入的 CT　图像，输入的影像文件，或者　numpy 矩阵（如果是numpy 矩阵，则需要传递 spacing 和origin 两个参数。
    # # @param pt 是 输入的 PET 图像，输入的影像文件，或者　numpy 矩阵（如果是numpy 矩阵，则需要传递 spacing 和origin 两个参数。
    # # @param ct_spacing 手动提供 ct 的 spacing
    # # @param ct_origin  手动提供 ct 的 origin
    # # @param pt_spacing 手动提供 pt 的 spacing
    # # @param pt_origin  手动提供 pt 的 origin
    # # """
    #
    #
    # def test_RandomGaussian(self):
    #     """
    #     @brief 对图像做高斯噪音处理
    #
    #     @param vec: 3D numpy data
    #     """
    #
    #     result = RandomGaussian().set_params(vec='/Users/Captain/Desktop/ADNI_test_1.nii').run()\
    #         .get_result(file='ADNI_test_1_RandomGaussian', itype='nii', dtype=np.int16) \
    #         .with_succ(self.succ).with_fail(self.fail)   # so slow
    #
    #
    # def test_Resample(self):   # 维度数值发生变化
    #     """
    #     @brief 使用插值器重新采样，按照新的间距、或者说尺寸进行重新采样。重采样的方法包括线性法、最邻近法等
    #
    #     @param input_data                 输入的待处理矩阵          ndarray类型，    默认值为    None
    #     @param channels                   矩阵重采样后的大小，       list，           默认值为    None
    #     @param zoom_num                   矩阵重采样后缩放的倍数     float           默认值为    1.
    #     @param model                      合并的模式，              int，            默认值为    3
    #     当model=0时，表示最邻近插值、
    #     当model=1时，表示样条插值、
    #     当model=2时，表示双线性插值、
    #     当model=3时，表示三次插值法（cubic interpolation））
    #     当model=4时，表示四次样条插值
    #     当model=5时，表示五次样条插值
    #
    #     当zoom_num与channels同时存在时，以channels为准，即channels的优先级更高。
    #     """
    #     zoom_num, model = 2., 2
    #     result = Resample().set_params(input_data='/Users/Captain/Desktop/ADNI_test_1.nii') \
    #             .set_params(zoom_num=zoom_num, model=model).run()\
    #             .get_result(file = 'ADNI_test_1_Resample', itype = 'nii', dtype = np.int16)\
    #             .with_succ(self.succ).with_fail(self.fail)
    #
    # def test_Rescale(self):
    #     """
    #     @brief 核心函数处理
    #
    #     @param in_upper  输入上线阈值， 浮点数，默认值为 None
    #     @param in_lower  输入下线阈值， 浮点数，默认值为 None
    #     @param out_upper 输出上线阈值， 浮点数，默认值为 255
    #     @param out_lower 输出下线阈值， 浮点数，默认值为 0
    #     @param vec       numpy 矩阵
    #     """
    #
    #     in_lower, in_upper = 0.1, 300
    #     out_lower, out_upper = 1.0, 300
    #     result = Rescale().set_params(vec = '/Users/Captain/Desktop/ADNI_test_1.nii')\
    #         .set_params(in_lower = in_lower, out_lower = out_lower, in_upper = in_upper, out_upper = out_upper).run()\
    #         .get_result(file = 'ADNI_test_1_Rescale', itype = 'nii', dtype = np.int16)\
    #         .with_succ(self.succ).with_fail(self.fail)

    #
    # def test_RG(self):  # 报错  # 仅支持二维输入，不支持nii多维数据
    #     """
    #     @brief 对一张PET图像进行粗糙分割（前景提取），返回一个0-1矩阵：背景的label为1，RIO的label为0
    #     @brief 主要是作为RW工具的轮廓约束
    #     @param thresh为区域生长的惩罚系数：惩罚系数的值越大，生长的难度就越大，默认值为120
    #     @param
    #     """
    #     thresh = 100
    #     result = RG().set_params(vec = '/Users/Captain/Desktop/ADNI_test_1.nii')\
    #         .set_params(thresh = thresh).run()\
    #         .get_result(file = 'ADNI_test_1_RG', itype = 'nii', dtype = np.int16)\
    #         .with_succ(self.succ).with_fail(self.fail)

    #
    # def test_RW_PET(self):  # so slow # 猜测：result数值极小，未能看到可视化
    #     """
    #         @brief 将指定的切片进行PET图像病灶轮廓的预测
    #         @param index   一个一维数组，存储用户指定的切片序列号:序列号是从 1 开始
    #         @param index:用户输入的切片的序列号，默认为一个空数组: np.array([])
    #         @param vec: numpy矩阵(3D)
    #     """
    #     index = np.arange(1,200,5)
    #     result = RW_PET().set_params(vec = '/Users/Captain/Desktop/ADNI_test_1.nii')\
    #         .set_params(index = index).run()\
    #         .get_result(file = 'ADNI_test_1_RW_PET', itype = 'nii', dtype = np.int16)\
    #         .with_succ(self.succ).with_fail(self.fail)

    #
    # def test_SelectChannels(self):
    #     """
    #     @brief 核心函数，主要用于选择通道以及相应的数据进行输出
    #
    #     @param input_data                 输入的待处理矩阵，           ndarray类型，    默认值为    None
    #     @param channels                   待选择的通道，              list，          默认值为    None
    #     @param output_data                输出参数（通道选择后的矩阵）  ndarray类型
    #
    #     """
    #     channels = list(range(0,150,2))
    #     result = SelectChannels().set_params(input_data = '/Users/Captain/Desktop/ADNI_test_1.nii')\
    #         .set_params(channels=channels).run()\
    #         .get_result(file = 'ADNI_test_1_SelectChannels', itype = 'nii', dtype = np.int16)\
    #         .with_succ(self.succ).with_fail(self.fail)

    #
    # def test_SliceFitting(self):   # ✔️
    #     """
    #     @brief 根据已有的等间隔标注的切片轮廓，得到中间未标注的切片的轮廓
    #     @brief 采用每隔 3 张标注一张的方法实现。最终得到的数量是 标注切片数量的 3倍
    #     @brief 标注示例：1，4，7，10，13... 抽取出来的切片还会保留在最终的序列中
    #
    #     @param vec_labeled 已经标注的切片的 numpy矩阵
    #     """
    #     result = SliceFitting().set_params(vec = '/Users/Captain/Desktop/ADNI_test_1.nii').run()\
    #         .get_result(file = 'ADNI_test_1_SliceFitting', itype = 'nii', dtype = np.int16)\
    #         .with_succ(self.succ).with_fail(self.fail)
    #
    #
    # # def test_SplitData(self):  # 无label数据，暂无法实例测试
    # #     """
    # #     @brief 将一张切片（图像）nparray按照对应的label分割成对应的多个矩阵：
    # #     ex:一张胸腔中已经将肝、肺、心脏按照label(1,2,3)对应标记，则将原始图像分割成三张切片：
    # #     只包含肝区域的矩阵（其他部分标为背景：0）、只包含肺区域的矩阵、只包含心脏区域的切片
    # #
    # #     默认：RIO区域使用1，2，3,4...标记，背景区域使用 0 标记
    # #
    # #     @brief parama vec_array     输入原始的切片矩阵
    # #     @brief parama label_array   输入对应的label矩阵
    # #     @brief parama label_splits  输入想要分割出来的label区域：使用列表存储，可以选择一次分割多个区域，deepneuro中是[1，2，3，4}
    # #     """
    # #     label_splits = [1]
    # #     label_array = 'xxx_label_array'
    # #     result = SplitData().set_params(vec_array = '/Users/Captain/Desktop/ADNI_test_1.nii')\
    # #         .set_params(label_splits = label_splits, label_array = label_array).run()\
    # #         .get_result(file = 'ADNI_test_1_SplitData', itype = 'nii', dtype = np.int16)\
    # #         .with_succ(self.succ).with_fail(self.fail)
    #
    #
    # def test_ThresholdProcessing(self): # ✔️
    #     """
    #     @brief 重新调整参数
    #
    #     @param input_upper 输入上限阈值，浮点数，默认值为None
    #     @param input_lower 输入下限阈值，浮点数，默认值为None
    #     @param vec         numpy矩阵
    #
    #     """
    #     input_lower = 2.0
    #     input_upper = 200.0
    #     result = ThresholdProcessing().set_params(vec = '/Users/Captain/Desktop/ADNI_test_1.nii')\
    #         .set_params(input_lower = input_lower, input_upper = input_upper).run()\
    #         .get_result(file = 'ADNI_test_1_ThresholdProcessing', itype = 'nii', dtype = np.int16)\
    #         .with_succ(self.succ).with_fail(self.fail)
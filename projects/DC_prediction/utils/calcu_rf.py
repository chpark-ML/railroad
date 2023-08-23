import numpy as np

""" code for calculating the receptive field size of featuremaps"""
input_size = [2000]

list_conv_kernel = [5, 9, 9, 9, 9, 9, 9] # 
list_conv_stride = [1, 1, 2, 2, 2, 2, 2]
list_conv_padding = [2, 8, 8, 8, 8, 8, 8]


list_rf = []
list_rf_diff = []
list_feature_dim = []
for i in range(len(list_conv_kernel)):
    if len(list_rf) == 0:
        list_rf.append(list_conv_kernel[i])
        list_rf_diff.append(list_conv_stride[i])
        list_feature_dim.append(
            [(i_size + 2*list_conv_padding[i] - list_conv_kernel[i]) // list_conv_stride[i] + 1 for _, i_size in enumerate(input_size)]
        ) 
    else:
        current_rf = list_rf[i-1] + (list_rf_diff[i-1] * (list_conv_kernel[i] - 1))
        current_rf_diff = list_rf_diff[i-1] * list_conv_stride[i]

        list_rf.append(current_rf)
        list_rf_diff.append(current_rf_diff)
        list_feature_dim.append(
            [(i_size + 2*list_conv_padding[i] - list_conv_kernel[i]) // list_conv_stride[i] + 1 for _, i_size in enumerate(list_feature_dim[-1])]
        ) 


print(list_rf)
print(list_rf_diff)
print("Input size : {}".format(input_size))
print("FeatureMap size : {}".format(list_feature_dim[-1]))
print("finished")
print()

from numpy import cumsum
from keras.layers import Conv2D, Lambda, Add, ZeroPadding2D

def Dilated2D(filters, kernel_size, dilation_rate=(1, 1), use_bias=True, **kwargs):
    """
    Support list of different dilation rates on first dimension(height).

    With zero padding on first dimension.
    
    # Arguments

    filters: int, the dimensionality of the output space  
        (i.e. the number of output filters in the convolution).
    kernel_size: (int, int), filter
        specifying the height and width of the 2D convolution window.  
    dilation_rate: (int, int) or (List[int], int), 
        specifying the dilation rate to use for dilated convolution.
        If first dimension is a tuple/list, length of it should be 
        equal to kernel_size[0] - 1
    """

    # Condition 1: dilation_rate = (a, k)
    if isinstance(dilation_rate[0], int):
        pad = (kernel_size[0] - 1) * dilation_rate[0]
        
        def BottomPaddingConv2D(inputs):
            # Bottom zero padding
            padded = ZeroPadding2D( padding=((0, pad), (0, 0)) )(inputs)

            conv = Conv2D(filters=filters, 
                          kernel_size=kernel_size, 
                          dilation_rate=dilation_rate,
                          use_bias=use_bias,
                          name='DL',
                          **kwargs
                          )(padded)
            return conv

        return BottomPaddingConv2D

    # Condition 2: dilation_rate = ( (a,b,c...), k )
    else:
        assert len(dilation_rate[0]) == kernel_size[0] - 1

        columns, column_rate = kernel_size[1], dilation_rate[1]
        dilation_index = cumsum(dilation_rate[0])

        def Sum_Conv2D(inputs):
            rows = [ 
                     Conv2D(filters=filters, 
                            kernel_size=(1, columns), 
                            dilation_rate=(1, column_rate), 
                            use_bias=use_bias,
                            name='DL_0',
                            **kwargs
                            )(inputs)
                    ]
            for i, ind in enumerate(dilation_index, start=1):
                # input_shape is `channel last`, (batch_size, height, width, channels)
                
                # drop values before target index
                drophead = Lambda(lambda tensor: tensor[:, ind:])(inputs)
                
                conv = Conv2D(filters=filters, 
                              kernel_size=(1, columns), 
                              dilation_rate=(1, column_rate), 
                              use_bias=False,
                              name=f'DL_{i}',
                              **kwargs
                              )(drophead)
                # Bottom zero padding
                conv = ZeroPadding2D( padding=((0, ind), (0, 0)) )(conv)
                rows.append(conv)

            return Add()(rows)

        return Sum_Conv2D
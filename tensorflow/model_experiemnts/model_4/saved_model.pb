��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
~
Adam/v/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_6/bias
w
'Adam/v/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_6/bias
w
'Adam/m/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/v/dense_6/kernel
�
)Adam/v/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/m/dense_6/kernel
�
)Adam/m/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/kernel*
_output_shapes
:	�*
dtype0
}
Adam/v/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/v/conv1d/bias
v
&Adam/v/conv1d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1d/bias*
_output_shapes	
:�*
dtype0
}
Adam/m/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/m/conv1d/bias
v
&Adam/m/conv1d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1d/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/conv1d/kernel
�
(Adam/v/conv1d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1d/kernel*#
_output_shapes
:�*
dtype0
�
Adam/m/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/conv1d/kernel
�
(Adam/m/conv1d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1d/kernel*#
_output_shapes
:�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	�*
dtype0
o
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv1d/bias
h
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes	
:�*
dtype0
{
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv1d/kernel
t
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*#
_output_shapes
:�*
dtype0
�
serving_default_lambda_1_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lambda_1_inputconv1d/kernelconv1d/biasdense_6/kerneldense_6/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_119566

NoOpNoOp
�"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�"
value�"B�" B�"
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias*
 
0
1
"2
#3*
 
0
1
"2
#3*
* 
�
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*

)trace_0
*trace_1* 

+trace_0
,trace_1* 
* 
�
-
_variables
._iterations
/_learning_rate
0_index_dict
1
_momentums
2_velocities
3_update_step_xla*

4serving_default* 
* 
* 
* 
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

:trace_0
;trace_1* 

<trace_0
=trace_1* 

0
1*

0
1*
* 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

"0
#1*

"0
#1*
* 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

L0*
* 
* 
* 
* 
* 
* 
C
.0
M1
N2
O3
P4
Q5
R6
S7
T8*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
M0
O1
Q2
S3*
 
N0
P1
R2
T3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
U	variables
V	keras_api
	Wtotal
	Xcount*
_Y
VARIABLE_VALUEAdam/m/conv1d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv1d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv1d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv1d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_6/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_6/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_6/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_6/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*

W0
X1*

U	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasdense_6/kerneldense_6/bias	iterationlearning_rateAdam/m/conv1d/kernelAdam/v/conv1d/kernelAdam/m/conv1d/biasAdam/v/conv1d/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biastotalcountConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_119772
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasdense_6/kerneldense_6/bias	iterationlearning_rateAdam/m/conv1d/kernelAdam/v/conv1d/kernelAdam/m/conv1d/biasAdam/v/conv1d/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biastotalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_119829��
�	
�
$__inference_signature_wrapper_119566
lambda_1_input
unknown:�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_119406s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name119562:&"
 
_user_specified_name119560:&"
 
_user_specified_name119558:&"
 
_user_specified_name119556:W S
'
_output_shapes
:���������
(
_user_specified_namelambda_1_input
�
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_119582

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:���������_
IdentityIdentityExpandDims:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_119483

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:���������_
IdentityIdentityExpandDims:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_6_layer_call_and_return_conditional_losses_119654

inputs4
!tensordot_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_conv1d_layer_call_and_return_conditional_losses_119433

inputsB
+conv1d_expanddims_1_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       _
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsPad:output:0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
__inference__traced_save_119772
file_prefix;
$read_disablecopyonread_conv1d_kernel:�3
$read_1_disablecopyonread_conv1d_bias:	�:
'read_2_disablecopyonread_dense_6_kernel:	�3
%read_3_disablecopyonread_dense_6_bias:,
"read_4_disablecopyonread_iteration:	 0
&read_5_disablecopyonread_learning_rate: D
-read_6_disablecopyonread_adam_m_conv1d_kernel:�D
-read_7_disablecopyonread_adam_v_conv1d_kernel:�:
+read_8_disablecopyonread_adam_m_conv1d_bias:	�:
+read_9_disablecopyonread_adam_v_conv1d_bias:	�B
/read_10_disablecopyonread_adam_m_dense_6_kernel:	�B
/read_11_disablecopyonread_adam_v_dense_6_kernel:	�;
-read_12_disablecopyonread_adam_m_dense_6_bias:;
-read_13_disablecopyonread_adam_v_dense_6_bias:)
read_14_disablecopyonread_total: )
read_15_disablecopyonread_count: 
savev2_const
identity_33��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv1d_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv1d_kernel^Read/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�*
dtype0n
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�f

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*#
_output_shapes
:�x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv1d_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv1d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_6_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_6_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_iteration^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	e

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: [

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_learning_rate^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_6/DisableCopyOnReadDisableCopyOnRead-read_6_disablecopyonread_adam_m_conv1d_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp-read_6_disablecopyonread_adam_m_conv1d_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�*
dtype0s
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�j
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*#
_output_shapes
:��
Read_7/DisableCopyOnReadDisableCopyOnRead-read_7_disablecopyonread_adam_v_conv1d_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp-read_7_disablecopyonread_adam_v_conv1d_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�*
dtype0s
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�j
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*#
_output_shapes
:�
Read_8/DisableCopyOnReadDisableCopyOnRead+read_8_disablecopyonread_adam_m_conv1d_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp+read_8_disablecopyonread_adam_m_conv1d_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_9/DisableCopyOnReadDisableCopyOnRead+read_9_disablecopyonread_adam_v_conv1d_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp+read_9_disablecopyonread_adam_v_conv1d_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_10/DisableCopyOnReadDisableCopyOnRead/read_10_disablecopyonread_adam_m_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp/read_10_disablecopyonread_adam_m_dense_6_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_11/DisableCopyOnReadDisableCopyOnRead/read_11_disablecopyonread_adam_v_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp/read_11_disablecopyonread_adam_v_dense_6_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_12/DisableCopyOnReadDisableCopyOnRead-read_12_disablecopyonread_adam_m_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp-read_12_disablecopyonread_adam_m_dense_6_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_13/DisableCopyOnReadDisableCopyOnRead-read_13_disablecopyonread_adam_v_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp-read_13_disablecopyonread_adam_v_dense_6_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_14/DisableCopyOnReadDisableCopyOnReadread_14_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpread_14_disablecopyonread_total^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_15/DisableCopyOnReadDisableCopyOnReadread_15_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpread_15_disablecopyonread_count^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_32Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_33IdentityIdentity_32:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_33Identity_33:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:%!

_user_specified_namecount:%!

_user_specified_nametotal:3/
-
_user_specified_nameAdam/v/dense_6/bias:3/
-
_user_specified_nameAdam/m/dense_6/bias:51
/
_user_specified_nameAdam/v/dense_6/kernel:51
/
_user_specified_nameAdam/m/dense_6/kernel:2
.
,
_user_specified_nameAdam/v/conv1d/bias:2	.
,
_user_specified_nameAdam/m/conv1d/bias:40
.
_user_specified_nameAdam/v/conv1d/kernel:40
.
_user_specified_nameAdam/m/conv1d/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_6/bias:.*
(
_user_specified_namedense_6/kernel:+'
%
_user_specified_nameconv1d/bias:-)
'
_user_specified_nameconv1d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
E
)__inference_lambda_1_layer_call_fn_119576

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_119483d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_6_layer_call_fn_119624

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_119468s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name119620:&"
 
_user_specified_name119618:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_conv1d_layer_call_fn_119597

inputs
unknown:�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_119433t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name119593:&"
 
_user_specified_name119591:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_4_layer_call_and_return_conditional_losses_119475
lambda_1_input$
conv1d_119434:�
conv1d_119436:	�!
dense_6_119469:	�
dense_6_119471:
identity��conv1d/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_119414�
conv1d/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_119434conv1d_119436*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_119433�
dense_6/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0dense_6_119469dense_6_119471*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_119468{
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������e
NoOpNoOp^conv1d/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:&"
 
_user_specified_name119471:&"
 
_user_specified_name119469:&"
 
_user_specified_name119436:&"
 
_user_specified_name119434:W S
'
_output_shapes
:���������
(
_user_specified_namelambda_1_input
�
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_119588

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:���������_
IdentityIdentityExpandDims:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�:
�
!__inference__wrapped_model_119406
lambda_1_inputQ
:model_4_conv1d_conv1d_expanddims_1_readvariableop_resource:�=
.model_4_conv1d_biasadd_readvariableop_resource:	�D
1model_4_dense_6_tensordot_readvariableop_resource:	�=
/model_4_dense_6_biasadd_readvariableop_resource:
identity��%model_4/conv1d/BiasAdd/ReadVariableOp�1model_4/conv1d/Conv1D/ExpandDims_1/ReadVariableOp�&model_4/dense_6/BiasAdd/ReadVariableOp�(model_4/dense_6/Tensordot/ReadVariableOpa
model_4/lambda_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model_4/lambda_1/ExpandDims
ExpandDimslambda_1_input(model_4/lambda_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:����������
model_4/conv1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
model_4/conv1d/PadPad$model_4/lambda_1/ExpandDims:output:0$model_4/conv1d/Pad/paddings:output:0*
T0*+
_output_shapes
:���������o
$model_4/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model_4/conv1d/Conv1D/ExpandDims
ExpandDimsmodel_4/conv1d/Pad:output:0-model_4/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
1model_4/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_4_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype0h
&model_4/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"model_4/conv1d/Conv1D/ExpandDims_1
ExpandDims9model_4/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model_4/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:��
model_4/conv1d/Conv1DConv2D)model_4/conv1d/Conv1D/ExpandDims:output:0+model_4/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
model_4/conv1d/Conv1D/SqueezeSqueezemodel_4/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

����������
%model_4/conv1d/BiasAdd/ReadVariableOpReadVariableOp.model_4_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_4/conv1d/BiasAddBiasAdd&model_4/conv1d/Conv1D/Squeeze:output:0-model_4/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������s
model_4/conv1d/ReluRelumodel_4/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:�����������
(model_4/dense_6/Tensordot/ReadVariableOpReadVariableOp1model_4_dense_6_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0h
model_4/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
model_4/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ~
model_4/dense_6/Tensordot/ShapeShape!model_4/conv1d/Relu:activations:0*
T0*
_output_shapes
::��i
'model_4/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"model_4/dense_6/Tensordot/GatherV2GatherV2(model_4/dense_6/Tensordot/Shape:output:0'model_4/dense_6/Tensordot/free:output:00model_4/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)model_4/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$model_4/dense_6/Tensordot/GatherV2_1GatherV2(model_4/dense_6/Tensordot/Shape:output:0'model_4/dense_6/Tensordot/axes:output:02model_4/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
model_4/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model_4/dense_6/Tensordot/ProdProd+model_4/dense_6/Tensordot/GatherV2:output:0(model_4/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!model_4/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
 model_4/dense_6/Tensordot/Prod_1Prod-model_4/dense_6/Tensordot/GatherV2_1:output:0*model_4/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%model_4/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model_4/dense_6/Tensordot/concatConcatV2'model_4/dense_6/Tensordot/free:output:0'model_4/dense_6/Tensordot/axes:output:0.model_4/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
model_4/dense_6/Tensordot/stackPack'model_4/dense_6/Tensordot/Prod:output:0)model_4/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
#model_4/dense_6/Tensordot/transpose	Transpose!model_4/conv1d/Relu:activations:0)model_4/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
!model_4/dense_6/Tensordot/ReshapeReshape'model_4/dense_6/Tensordot/transpose:y:0(model_4/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
 model_4/dense_6/Tensordot/MatMulMatMul*model_4/dense_6/Tensordot/Reshape:output:00model_4/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������k
!model_4/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:i
'model_4/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"model_4/dense_6/Tensordot/concat_1ConcatV2+model_4/dense_6/Tensordot/GatherV2:output:0*model_4/dense_6/Tensordot/Const_2:output:00model_4/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model_4/dense_6/TensordotReshape*model_4/dense_6/Tensordot/MatMul:product:0+model_4/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
&model_4/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_4_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_4/dense_6/BiasAddBiasAdd"model_4/dense_6/Tensordot:output:0.model_4/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������s
IdentityIdentity model_4/dense_6/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp&^model_4/conv1d/BiasAdd/ReadVariableOp2^model_4/conv1d/Conv1D/ExpandDims_1/ReadVariableOp'^model_4/dense_6/BiasAdd/ReadVariableOp)^model_4/dense_6/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2N
%model_4/conv1d/BiasAdd/ReadVariableOp%model_4/conv1d/BiasAdd/ReadVariableOp2f
1model_4/conv1d/Conv1D/ExpandDims_1/ReadVariableOp1model_4/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2P
&model_4/dense_6/BiasAdd/ReadVariableOp&model_4/dense_6/BiasAdd/ReadVariableOp2T
(model_4/dense_6/Tensordot/ReadVariableOp(model_4/dense_6/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
'
_output_shapes
:���������
(
_user_specified_namelambda_1_input
�	
�
(__inference_model_4_layer_call_fn_119522
lambda_1_input
unknown:�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_119496s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name119518:&"
 
_user_specified_name119516:&"
 
_user_specified_name119514:&"
 
_user_specified_name119512:W S
'
_output_shapes
:���������
(
_user_specified_namelambda_1_input
�	
�
(__inference_model_4_layer_call_fn_119509
lambda_1_input
unknown:�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_119475s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name119505:&"
 
_user_specified_name119503:&"
 
_user_specified_name119501:&"
 
_user_specified_name119499:W S
'
_output_shapes
:���������
(
_user_specified_namelambda_1_input
�
�
C__inference_dense_6_layer_call_and_return_conditional_losses_119468

inputs4
!tensordot_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�L
�	
"__inference__traced_restore_119829
file_prefix5
assignvariableop_conv1d_kernel:�-
assignvariableop_1_conv1d_bias:	�4
!assignvariableop_2_dense_6_kernel:	�-
assignvariableop_3_dense_6_bias:&
assignvariableop_4_iteration:	 *
 assignvariableop_5_learning_rate: >
'assignvariableop_6_adam_m_conv1d_kernel:�>
'assignvariableop_7_adam_v_conv1d_kernel:�4
%assignvariableop_8_adam_m_conv1d_bias:	�4
%assignvariableop_9_adam_v_conv1d_bias:	�<
)assignvariableop_10_adam_m_dense_6_kernel:	�<
)assignvariableop_11_adam_v_dense_6_kernel:	�5
'assignvariableop_12_adam_m_dense_6_bias:5
'assignvariableop_13_adam_v_dense_6_bias:#
assignvariableop_14_total: #
assignvariableop_15_count: 
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_6_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_6_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_iterationIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_learning_rateIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp'assignvariableop_6_adam_m_conv1d_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp'assignvariableop_7_adam_v_conv1d_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_m_conv1d_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_v_conv1d_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_adam_m_dense_6_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_v_dense_6_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp'assignvariableop_12_adam_m_dense_6_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp'assignvariableop_13_adam_v_dense_6_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_17IdentityIdentity_16:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_17Identity_17:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%!

_user_specified_namecount:%!

_user_specified_nametotal:3/
-
_user_specified_nameAdam/v/dense_6/bias:3/
-
_user_specified_nameAdam/m/dense_6/bias:51
/
_user_specified_nameAdam/v/dense_6/kernel:51
/
_user_specified_nameAdam/m/dense_6/kernel:2
.
,
_user_specified_nameAdam/v/conv1d/bias:2	.
,
_user_specified_nameAdam/m/conv1d/bias:40
.
_user_specified_nameAdam/v/conv1d/kernel:40
.
_user_specified_nameAdam/m/conv1d/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_6/bias:.*
(
_user_specified_namedense_6/kernel:+'
%
_user_specified_nameconv1d/bias:-)
'
_user_specified_nameconv1d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_119414

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:���������_
IdentityIdentityExpandDims:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_4_layer_call_and_return_conditional_losses_119496
lambda_1_input$
conv1d_119485:�
conv1d_119487:	�!
dense_6_119490:	�
dense_6_119492:
identity��conv1d/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_119483�
conv1d/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0conv1d_119485conv1d_119487*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_119433�
dense_6/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0dense_6_119490dense_6_119492*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_119468{
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������e
NoOpNoOp^conv1d/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:&"
 
_user_specified_name119492:&"
 
_user_specified_name119490:&"
 
_user_specified_name119487:&"
 
_user_specified_name119485:W S
'
_output_shapes
:���������
(
_user_specified_namelambda_1_input
�
E
)__inference_lambda_1_layer_call_fn_119571

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_119414d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_conv1d_layer_call_and_return_conditional_losses_119615

inputsB
+conv1d_expanddims_1_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       _
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsPad:output:0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
lambda_1_input7
 serving_default_lambda_1_input:0���������?
dense_64
StatefulPartitionedCall:0���������tensorflow/serving/predict:�e
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
<
0
1
"2
#3"
trackable_list_wrapper
<
0
1
"2
#3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�
)trace_0
*trace_12�
(__inference_model_4_layer_call_fn_119509
(__inference_model_4_layer_call_fn_119522�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z)trace_0z*trace_1
�
+trace_0
,trace_12�
C__inference_model_4_layer_call_and_return_conditional_losses_119475
C__inference_model_4_layer_call_and_return_conditional_losses_119496�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z+trace_0z,trace_1
�B�
!__inference__wrapped_model_119406lambda_1_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
-
_variables
._iterations
/_learning_rate
0_index_dict
1
_momentums
2_velocities
3_update_step_xla"
experimentalOptimizer
,
4serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
:trace_0
;trace_12�
)__inference_lambda_1_layer_call_fn_119571
)__inference_lambda_1_layer_call_fn_119576�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z:trace_0z;trace_1
�
<trace_0
=trace_12�
D__inference_lambda_1_layer_call_and_return_conditional_losses_119582
D__inference_lambda_1_layer_call_and_return_conditional_losses_119588�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z<trace_0z=trace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ctrace_02�
'__inference_conv1d_layer_call_fn_119597�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zCtrace_0
�
Dtrace_02�
B__inference_conv1d_layer_call_and_return_conditional_losses_119615�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zDtrace_0
$:"�2conv1d/kernel
:�2conv1d/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
Jtrace_02�
(__inference_dense_6_layer_call_fn_119624�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0
�
Ktrace_02�
C__inference_dense_6_layer_call_and_return_conditional_losses_119654�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zKtrace_0
!:	�2dense_6/kernel
:2dense_6/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
L0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_4_layer_call_fn_119509lambda_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_4_layer_call_fn_119522lambda_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_4_layer_call_and_return_conditional_losses_119475lambda_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_4_layer_call_and_return_conditional_losses_119496lambda_1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
_
.0
M1
N2
O3
P4
Q5
R6
S7
T8"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
<
M0
O1
Q2
S3"
trackable_list_wrapper
<
N0
P1
R2
T3"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_119566lambda_1_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_lambda_1_layer_call_fn_119571inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_lambda_1_layer_call_fn_119576inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_lambda_1_layer_call_and_return_conditional_losses_119582inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_lambda_1_layer_call_and_return_conditional_losses_119588inputs"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv1d_layer_call_fn_119597inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv1d_layer_call_and_return_conditional_losses_119615inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_6_layer_call_fn_119624inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_6_layer_call_and_return_conditional_losses_119654inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
U	variables
V	keras_api
	Wtotal
	Xcount"
_tf_keras_metric
):'�2Adam/m/conv1d/kernel
):'�2Adam/v/conv1d/kernel
:�2Adam/m/conv1d/bias
:�2Adam/v/conv1d/bias
&:$	�2Adam/m/dense_6/kernel
&:$	�2Adam/v/dense_6/kernel
:2Adam/m/dense_6/bias
:2Adam/v/dense_6/bias
.
W0
X1"
trackable_list_wrapper
-
U	variables"
_generic_user_object
:  (2total
:  (2count�
!__inference__wrapped_model_119406v"#7�4
-�*
(�%
lambda_1_input���������
� "5�2
0
dense_6%�"
dense_6����������
B__inference_conv1d_layer_call_and_return_conditional_losses_119615l3�0
)�&
$�!
inputs���������
� "1�.
'�$
tensor_0����������
� �
'__inference_conv1d_layer_call_fn_119597a3�0
)�&
$�!
inputs���������
� "&�#
unknown�����������
C__inference_dense_6_layer_call_and_return_conditional_losses_119654l"#4�1
*�'
%�"
inputs����������
� "0�-
&�#
tensor_0���������
� �
(__inference_dense_6_layer_call_fn_119624a"#4�1
*�'
%�"
inputs����������
� "%�"
unknown����������
D__inference_lambda_1_layer_call_and_return_conditional_losses_119582k7�4
-�*
 �
inputs���������

 
p
� "0�-
&�#
tensor_0���������
� �
D__inference_lambda_1_layer_call_and_return_conditional_losses_119588k7�4
-�*
 �
inputs���������

 
p 
� "0�-
&�#
tensor_0���������
� �
)__inference_lambda_1_layer_call_fn_119571`7�4
-�*
 �
inputs���������

 
p
� "%�"
unknown����������
)__inference_lambda_1_layer_call_fn_119576`7�4
-�*
 �
inputs���������

 
p 
� "%�"
unknown����������
C__inference_model_4_layer_call_and_return_conditional_losses_119475y"#?�<
5�2
(�%
lambda_1_input���������
p

 
� "0�-
&�#
tensor_0���������
� �
C__inference_model_4_layer_call_and_return_conditional_losses_119496y"#?�<
5�2
(�%
lambda_1_input���������
p 

 
� "0�-
&�#
tensor_0���������
� �
(__inference_model_4_layer_call_fn_119509n"#?�<
5�2
(�%
lambda_1_input���������
p

 
� "%�"
unknown����������
(__inference_model_4_layer_call_fn_119522n"#?�<
5�2
(�%
lambda_1_input���������
p 

 
� "%�"
unknown����������
$__inference_signature_wrapper_119566�"#I�F
� 
?�<
:
lambda_1_input(�%
lambda_1_input���������"5�2
0
dense_6%�"
dense_6���������
са
Ќу
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
ђ
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceѕ
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
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
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
output"out_typeіьout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements(
handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
ѕ"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628┬Р
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
Adam/v/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_7/bias
w
'Adam/v/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_7/bias
w
'Adam/m/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/bias*
_output_shapes
:*
dtype0
Є
Adam/v/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*&
shared_nameAdam/v/dense_7/kernel
ђ
)Adam/v/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/kernel*
_output_shapes
:	ђ*
dtype0
Є
Adam/m/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*&
shared_nameAdam/m/dense_7/kernel
ђ
)Adam/m/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/kernel*
_output_shapes
:	ђ*
dtype0
Ї
Adam/v/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_nameAdam/v/lstm/lstm_cell/bias
є
.Adam/v/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
Ї
Adam/m/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_nameAdam/m/lstm/lstm_cell/bias
є
.Adam/m/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
ф
&Adam/v/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*7
shared_name(&Adam/v/lstm/lstm_cell/recurrent_kernel
Б
:Adam/v/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp&Adam/v/lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
ђђ*
dtype0
ф
&Adam/m/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*7
shared_name(&Adam/m/lstm/lstm_cell/recurrent_kernel
Б
:Adam/m/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp&Adam/m/lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
ђђ*
dtype0
Ћ
Adam/v/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*-
shared_nameAdam/v/lstm/lstm_cell/kernel
ј
0Adam/v/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm/lstm_cell/kernel*
_output_shapes
:	ђ*
dtype0
Ћ
Adam/m/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*-
shared_nameAdam/m/lstm/lstm_cell/kernel
ј
0Adam/m/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm/lstm_cell/kernel*
_output_shapes
:	ђ*
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

lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*$
shared_namelstm/lstm_cell/bias
x
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes	
:ђ*
dtype0
ю
lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*0
shared_name!lstm/lstm_cell/recurrent_kernel
Ћ
3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
ђђ*
dtype0
Є
lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*&
shared_namelstm/lstm_cell/kernel
ђ
)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes
:	ђ*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	ђ*
dtype0
z
serving_default_input_2Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
Д
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2lstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biasdense_7/kerneldense_7/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_209355

NoOpNoOp
║,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ш+
valueв+BУ+ Bр+
┤
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
ј
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
┴
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
д
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
'
%0
&1
'2
#3
$4*
'
%0
&1
'2
#3
$4*
* 
░
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*

-trace_0
.trace_1* 

/trace_0
0trace_1* 
* 
Ђ
1
_variables
2_iterations
3_learning_rate
4_index_dict
5
_momentums
6_velocities
7_update_step_xla*

8serving_default* 
* 
* 
* 
Љ
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

>trace_0
?trace_1* 

@trace_0
Atrace_1* 

%0
&1
'2*

%0
&1
'2*
* 
Ъ

Bstates
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Htrace_0
Itrace_1
Jtrace_2
Ktrace_3* 
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
* 
с
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
V_random_generator
W
state_size

%kernel
&recurrent_kernel
'bias*
* 

#0
$1*

#0
$1*
* 
Њ
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUElstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

_0*
* 
* 
* 
* 
* 
* 
R
20
`1
a2
b3
c4
d5
e6
f7
g8
h9
i10*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
`0
b1
d2
f3
h4*
'
a0
c1
e2
g3
i4*
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

0*
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

%0
&1
'2*

%0
&1
'2*
* 
Њ
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

otrace_0
ptrace_1* 

qtrace_0
rtrace_1* 
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
s	variables
t	keras_api
	utotal
	vcount*
ga
VARIABLE_VALUEAdam/m/lstm/lstm_cell/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/lstm/lstm_cell/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/m/lstm/lstm_cell/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/v/lstm/lstm_cell/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/lstm/lstm_cell/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/lstm/lstm_cell/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_7/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_7/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_7/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_7/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 

u0
v1*

s	variables*
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
П
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_7/kerneldense_7/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/bias	iterationlearning_rateAdam/m/lstm/lstm_cell/kernelAdam/v/lstm/lstm_cell/kernel&Adam/m/lstm/lstm_cell/recurrent_kernel&Adam/v/lstm/lstm_cell/recurrent_kernelAdam/m/lstm/lstm_cell/biasAdam/v/lstm/lstm_cell/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biastotalcountConst* 
Tin
2*
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
GPU 2J 8ѓ *(
f#R!
__inference__traced_save_210254
п
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_7/kerneldense_7/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/bias	iterationlearning_rateAdam/m/lstm/lstm_cell/kernelAdam/v/lstm/lstm_cell/kernel&Adam/m/lstm/lstm_cell/recurrent_kernel&Adam/v/lstm/lstm_cell/recurrent_kernelAdam/m/lstm/lstm_cell/biasAdam/v/lstm/lstm_cell/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biastotalcount*
Tin
2*
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
GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_210320т■
к	
├
while_cond_208793
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_208793___redundant_placeholder04
0while_while_cond_208793___redundant_placeholder14
0while_while_cond_208793___redundant_placeholder24
0while_while_cond_208793___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ђ:         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
№
ќ
(__inference_dense_7_layer_call_fn_210010

inputs
unknown:	ђ
	unknown_0:
identityѕбStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_209091o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name210006:&"
 
_user_specified_name210004:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ю:
и
while_body_209481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђF
2while_lstm_cell_matmul_1_readvariableop_resource_0:
ђђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђD
0while_lstm_cell_matmul_1_readvariableop_resource:
ђђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђю
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:         ђЂ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ђo
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:         ђљ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђЁ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:         ђl
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђћ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ђw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ђБ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ђ:         ђ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ю:
и
while_body_209771
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђF
2while_lstm_cell_matmul_1_readvariableop_resource_0:
ђђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђD
0while_lstm_cell_matmul_1_readvariableop_resource:
ђђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђю
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:         ђЂ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ђo
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:         ђљ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђЁ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:         ђl
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђћ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ђw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ђБ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ђ:         ђ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ц
E
)__inference_lambda_2_layer_call_fn_209365

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_209106d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
УJ
Є
@__inference_lstm_layer_call_and_return_conditional_losses_209074

inputs;
(lstm_cell_matmul_readvariableop_resource:	ђ>
*lstm_cell_matmul_1_readvariableop_resource:
ђђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђј
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:         ђr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:         ђ~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:         ђ`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђѓ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ђ:         ђ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_208989*
condR
while_cond_208988*M
output_shapes<
:: : : : :         ђ:         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ђ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ђ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ђЊ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ІK
Ѕ
@__inference_lstm_layer_call_and_return_conditional_losses_209711
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	ђ>
*lstm_cell_matmul_1_readvariableop_resource:
ђђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђј
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:         ђr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:         ђ~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:         ђ`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђѓ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ђ:         ђ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_209626*
condR
while_cond_209625*M
output_shapes<
:: : : : :         ђ:         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ђ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ђ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ђЊ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
Я
х
%__inference_lstm_layer_call_fn_209388
inputs_0
unknown:	ђ
	unknown_0:
ђђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_208717p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name209384:&"
 
_user_specified_name209382:&"
 
_user_specified_name209380:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
Ц
E
)__inference_lambda_2_layer_call_fn_209360

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_208928d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┴
ѓ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_208632

inputs

states
states_11
matmul_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ђW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ђV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ђO
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ђ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ђU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ђW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ђL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ђd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ђY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ђ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ђ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ђm
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ђ:         ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:PL
(
_output_shapes
:         ђ
 
_user_specified_namestates:PL
(
_output_shapes
:         ђ
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к	
├
while_cond_209770
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_209770___redundant_placeholder04
0while_while_cond_209770___redundant_placeholder14
0while_while_cond_209770___redundant_placeholder24
0while_while_cond_209770___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ђ:         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
УJ
Є
@__inference_lstm_layer_call_and_return_conditional_losses_209856

inputs;
(lstm_cell_matmul_readvariableop_resource:	ђ>
*lstm_cell_matmul_1_readvariableop_resource:
ђђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђј
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:         ђr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:         ђ~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:         ђ`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђѓ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ђ:         ђ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_209771*
condR
while_cond_209770*M
output_shapes<
:: : : : :         ђ:         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ђ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ђ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ђЊ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
В	
Ь
(__inference_model_5_layer_call_fn_209281
input_2
unknown:	ђ
	unknown_0:
ђђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_209098o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name209277:&"
 
_user_specified_name209275:&"
 
_user_specified_name209273:&"
 
_user_specified_name209271:&"
 
_user_specified_name209269:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
ю:
и
while_body_209916
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђF
2while_lstm_cell_matmul_1_readvariableop_resource_0:
ђђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђD
0while_lstm_cell_matmul_1_readvariableop_resource:
ђђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђю
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:         ђЂ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ђo
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:         ђљ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђЁ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:         ђl
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђћ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ђw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ђБ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ђ:         ђ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╚
│
%__inference_lstm_layer_call_fn_209410

inputs
unknown:	ђ
	unknown_0:
ђђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_209074p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name209406:&"
 
_user_specified_name209404:&"
 
_user_specified_name209402:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
к	
Ж
$__inference_signature_wrapper_209355
input_2
unknown:	ђ
	unknown_0:
ђђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:
identityѕбStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_208570o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name209351:&"
 
_user_specified_name209349:&"
 
_user_specified_name209347:&"
 
_user_specified_name209345:&"
 
_user_specified_name209343:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
к	
├
while_cond_209625
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_209625___redundant_placeholder04
0while_while_cond_209625___redundant_placeholder14
0while_while_cond_209625___redundant_placeholder24
0while_while_cond_209625___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ђ:         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ІK
Ѕ
@__inference_lstm_layer_call_and_return_conditional_losses_209566
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	ђ>
*lstm_cell_matmul_1_readvariableop_resource:
ђђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђј
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:         ђr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:         ђ~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:         ђ`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђѓ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ђ:         ђ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_209481*
condR
while_cond_209480*M
output_shapes<
:: : : : :         ђ:         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ђ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ђ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ђЊ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
ж%
л
while_body_208794
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_208818_0:	ђ,
while_lstm_cell_208820_0:
ђђ'
while_lstm_cell_208822_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_208818:	ђ*
while_lstm_cell_208820:
ђђ%
while_lstm_cell_208822:	ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Д
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_208818_0while_lstm_cell_208820_0while_lstm_cell_208822_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_208779r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ђ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:00while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ј
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ђј
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         ђR

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_208818while_lstm_cell_208818_0"2
while_lstm_cell_208820while_lstm_cell_208820_0"2
while_lstm_cell_208822while_lstm_cell_208822_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ђ:         ђ: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name208822:&	"
 
_user_specified_name208820:&"
 
_user_specified_name208818:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╚
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_208928

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
:         _
IdentityIdentityExpandDims:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ќ
┬
C__inference_model_5_layer_call_and_return_conditional_losses_209266
input_2
lstm_209253:	ђ
lstm_209255:
ђђ
lstm_209257:	ђ!
dense_7_209260:	ђ
dense_7_209262:
identityѕбdense_7/StatefulPartitionedCallбlstm/StatefulPartitionedCallй
lambda_2/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_209106І
lstm/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0lstm_209253lstm_209255lstm_209257*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_209252І
dense_7/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_7_209260dense_7_209262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_209091w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         c
NoOpNoOp ^dense_7/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:&"
 
_user_specified_name209262:&"
 
_user_specified_name209260:&"
 
_user_specified_name209257:&"
 
_user_specified_name209255:&"
 
_user_specified_name209253:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
к	
├
while_cond_208988
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_208988___redundant_placeholder04
0while_while_cond_208988___redundant_placeholder14
0while_while_cond_208988___redundant_placeholder24
0while_while_cond_208988___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ђ:         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
зџ
ѓ
__inference__traced_save_210254
file_prefix8
%read_disablecopyonread_dense_7_kernel:	ђ3
%read_1_disablecopyonread_dense_7_bias:A
.read_2_disablecopyonread_lstm_lstm_cell_kernel:	ђL
8read_3_disablecopyonread_lstm_lstm_cell_recurrent_kernel:
ђђ;
,read_4_disablecopyonread_lstm_lstm_cell_bias:	ђ,
"read_5_disablecopyonread_iteration:	 0
&read_6_disablecopyonread_learning_rate: H
5read_7_disablecopyonread_adam_m_lstm_lstm_cell_kernel:	ђH
5read_8_disablecopyonread_adam_v_lstm_lstm_cell_kernel:	ђS
?read_9_disablecopyonread_adam_m_lstm_lstm_cell_recurrent_kernel:
ђђT
@read_10_disablecopyonread_adam_v_lstm_lstm_cell_recurrent_kernel:
ђђC
4read_11_disablecopyonread_adam_m_lstm_lstm_cell_bias:	ђC
4read_12_disablecopyonread_adam_v_lstm_lstm_cell_bias:	ђB
/read_13_disablecopyonread_adam_m_dense_7_kernel:	ђB
/read_14_disablecopyonread_adam_v_dense_7_kernel:	ђ;
-read_15_disablecopyonread_adam_m_dense_7_bias:;
-read_16_disablecopyonread_adam_v_dense_7_bias:)
read_17_disablecopyonread_total: )
read_18_disablecopyonread_count: 
savev2_const
identity_39ѕбMergeV2CheckpointsбRead/DisableCopyOnReadбRead/ReadVariableOpбRead_1/DisableCopyOnReadбRead_1/ReadVariableOpбRead_10/DisableCopyOnReadбRead_10/ReadVariableOpбRead_11/DisableCopyOnReadбRead_11/ReadVariableOpбRead_12/DisableCopyOnReadбRead_12/ReadVariableOpбRead_13/DisableCopyOnReadбRead_13/ReadVariableOpбRead_14/DisableCopyOnReadбRead_14/ReadVariableOpбRead_15/DisableCopyOnReadбRead_15/ReadVariableOpбRead_16/DisableCopyOnReadбRead_16/ReadVariableOpбRead_17/DisableCopyOnReadбRead_17/ReadVariableOpбRead_18/DisableCopyOnReadбRead_18/ReadVariableOpбRead_2/DisableCopyOnReadбRead_2/ReadVariableOpбRead_3/DisableCopyOnReadбRead_3/ReadVariableOpбRead_4/DisableCopyOnReadбRead_4/ReadVariableOpбRead_5/DisableCopyOnReadбRead_5/ReadVariableOpбRead_6/DisableCopyOnReadбRead_6/ReadVariableOpбRead_7/DisableCopyOnReadбRead_7/ReadVariableOpбRead_8/DisableCopyOnReadбRead_8/ReadVariableOpбRead_9/DisableCopyOnReadбRead_9/ReadVariableOpw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_dense_7_kernel"/device:CPU:0*
_output_shapes
 б
Read/ReadVariableOpReadVariableOp%read_disablecopyonread_dense_7_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђb

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђy
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_dense_7_bias"/device:CPU:0*
_output_shapes
 А
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_dense_7_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:ѓ
Read_2/DisableCopyOnReadDisableCopyOnRead.read_2_disablecopyonread_lstm_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 »
Read_2/ReadVariableOpReadVariableOp.read_2_disablecopyonread_lstm_lstm_cell_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђd

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђї
Read_3/DisableCopyOnReadDisableCopyOnRead8read_3_disablecopyonread_lstm_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ║
Read_3/ReadVariableOpReadVariableOp8read_3_disablecopyonread_lstm_lstm_cell_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ђђ*
dtype0o

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђe

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђђ
Read_4/DisableCopyOnReadDisableCopyOnRead,read_4_disablecopyonread_lstm_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Е
Read_4/ReadVariableOpReadVariableOp,read_4_disablecopyonread_lstm_lstm_cell_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђ`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђv
Read_5/DisableCopyOnReadDisableCopyOnRead"read_5_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 џ
Read_5/ReadVariableOpReadVariableOp"read_5_disablecopyonread_iteration^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 ъ
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_learning_rate^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: Ѕ
Read_7/DisableCopyOnReadDisableCopyOnRead5read_7_disablecopyonread_adam_m_lstm_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 Х
Read_7/ReadVariableOpReadVariableOp5read_7_disablecopyonread_adam_m_lstm_lstm_cell_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђf
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђЅ
Read_8/DisableCopyOnReadDisableCopyOnRead5read_8_disablecopyonread_adam_v_lstm_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 Х
Read_8/ReadVariableOpReadVariableOp5read_8_disablecopyonread_adam_v_lstm_lstm_cell_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђf
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђЊ
Read_9/DisableCopyOnReadDisableCopyOnRead?read_9_disablecopyonread_adam_m_lstm_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┴
Read_9/ReadVariableOpReadVariableOp?read_9_disablecopyonread_adam_m_lstm_lstm_cell_recurrent_kernel^Read_9/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ђђ*
dtype0p
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђg
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђЋ
Read_10/DisableCopyOnReadDisableCopyOnRead@read_10_disablecopyonread_adam_v_lstm_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ─
Read_10/ReadVariableOpReadVariableOp@read_10_disablecopyonread_adam_v_lstm_lstm_cell_recurrent_kernel^Read_10/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ђђ*
dtype0q
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђg
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђЅ
Read_11/DisableCopyOnReadDisableCopyOnRead4read_11_disablecopyonread_adam_m_lstm_lstm_cell_bias"/device:CPU:0*
_output_shapes
 │
Read_11/ReadVariableOpReadVariableOp4read_11_disablecopyonread_adam_m_lstm_lstm_cell_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђЅ
Read_12/DisableCopyOnReadDisableCopyOnRead4read_12_disablecopyonread_adam_v_lstm_lstm_cell_bias"/device:CPU:0*
_output_shapes
 │
Read_12/ReadVariableOpReadVariableOp4read_12_disablecopyonread_adam_v_lstm_lstm_cell_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђё
Read_13/DisableCopyOnReadDisableCopyOnRead/read_13_disablecopyonread_adam_m_dense_7_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_13/ReadVariableOpReadVariableOp/read_13_disablecopyonread_adam_m_dense_7_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђf
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђё
Read_14/DisableCopyOnReadDisableCopyOnRead/read_14_disablecopyonread_adam_v_dense_7_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_14/ReadVariableOpReadVariableOp/read_14_disablecopyonread_adam_v_dense_7_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0p
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђf
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђѓ
Read_15/DisableCopyOnReadDisableCopyOnRead-read_15_disablecopyonread_adam_m_dense_7_bias"/device:CPU:0*
_output_shapes
 Ф
Read_15/ReadVariableOpReadVariableOp-read_15_disablecopyonread_adam_m_dense_7_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:ѓ
Read_16/DisableCopyOnReadDisableCopyOnRead-read_16_disablecopyonread_adam_v_dense_7_bias"/device:CPU:0*
_output_shapes
 Ф
Read_16/ReadVariableOpReadVariableOp-read_16_disablecopyonread_adam_v_dense_7_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_17/DisableCopyOnReadDisableCopyOnReadread_17_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Ў
Read_17/ReadVariableOpReadVariableOpread_17_disablecopyonread_total^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_18/DisableCopyOnReadDisableCopyOnReadread_18_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Ў
Read_18/ReadVariableOpReadVariableOpread_18_disablecopyonread_count^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: ┬
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*в
valueрBяB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЋ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B ѕ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *"
dtypes
2	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_38Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_39IdentityIdentity_38:output:0^NoOp*
T0*
_output_shapes
: ј
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_39Identity_39:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2(
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
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:%!

_user_specified_namecount:%!

_user_specified_nametotal:3/
-
_user_specified_nameAdam/v/dense_7/bias:3/
-
_user_specified_nameAdam/m/dense_7/bias:51
/
_user_specified_nameAdam/v/dense_7/kernel:51
/
_user_specified_nameAdam/m/dense_7/kernel::6
4
_user_specified_nameAdam/v/lstm/lstm_cell/bias::6
4
_user_specified_nameAdam/m/lstm/lstm_cell/bias:FB
@
_user_specified_name(&Adam/v/lstm/lstm_cell/recurrent_kernel:F
B
@
_user_specified_name(&Adam/m/lstm/lstm_cell/recurrent_kernel:<	8
6
_user_specified_nameAdam/v/lstm/lstm_cell/kernel:<8
6
_user_specified_nameAdam/m/lstm/lstm_cell/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:3/
-
_user_specified_namelstm/lstm_cell/bias:?;
9
_user_specified_name!lstm/lstm_cell/recurrent_kernel:51
/
_user_specified_namelstm/lstm_cell/kernel:,(
&
_user_specified_namedense_7/bias:.*
(
_user_specified_namedense_7/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Щ	
ш
C__inference_dense_7_layer_call_and_return_conditional_losses_210020

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
─
З
*__inference_lstm_cell_layer_call_fn_210054

inputs
states_0
states_1
unknown:	ђ
	unknown_0:
ђђ
	unknown_1:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_208779p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ђ:         ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name210046:&"
 
_user_specified_name210044:&"
 
_user_specified_name210042:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_1:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ю:
и
while_body_209167
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђF
2while_lstm_cell_matmul_1_readvariableop_resource_0:
ђђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђD
0while_lstm_cell_matmul_1_readvariableop_resource:
ђђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђю
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:         ђЂ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ђo
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:         ђљ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђЁ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:         ђl
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђћ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ђw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ђБ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ђ:         ђ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
к	
├
while_cond_209480
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_209480___redundant_placeholder04
0while_while_cond_209480___redundant_placeholder14
0while_while_cond_209480___redundant_placeholder24
0while_while_cond_209480___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ђ:         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
УJ
Є
@__inference_lstm_layer_call_and_return_conditional_losses_210001

inputs;
(lstm_cell_matmul_readvariableop_resource:	ђ>
*lstm_cell_matmul_1_readvariableop_resource:
ђђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђј
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:         ђr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:         ђ~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:         ђ`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђѓ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ђ:         ђ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_209916*
condR
while_cond_209915*M
output_shapes<
:: : : : :         ђ:         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ђ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ђ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ђЊ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ў\
Ф
"__inference__traced_restore_210320
file_prefix2
assignvariableop_dense_7_kernel:	ђ-
assignvariableop_1_dense_7_bias:;
(assignvariableop_2_lstm_lstm_cell_kernel:	ђF
2assignvariableop_3_lstm_lstm_cell_recurrent_kernel:
ђђ5
&assignvariableop_4_lstm_lstm_cell_bias:	ђ&
assignvariableop_5_iteration:	 *
 assignvariableop_6_learning_rate: B
/assignvariableop_7_adam_m_lstm_lstm_cell_kernel:	ђB
/assignvariableop_8_adam_v_lstm_lstm_cell_kernel:	ђM
9assignvariableop_9_adam_m_lstm_lstm_cell_recurrent_kernel:
ђђN
:assignvariableop_10_adam_v_lstm_lstm_cell_recurrent_kernel:
ђђ=
.assignvariableop_11_adam_m_lstm_lstm_cell_bias:	ђ=
.assignvariableop_12_adam_v_lstm_lstm_cell_bias:	ђ<
)assignvariableop_13_adam_m_dense_7_kernel:	ђ<
)assignvariableop_14_adam_v_dense_7_kernel:	ђ5
'assignvariableop_15_adam_m_dense_7_bias:5
'assignvariableop_16_adam_v_dense_7_bias:#
assignvariableop_17_total: #
assignvariableop_18_count: 
identity_20ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9┼
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*в
valueрBяB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHў
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B ѓ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOpAssignVariableOpassignvariableop_dense_7_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_7_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_2AssignVariableOp(assignvariableop_2_lstm_lstm_cell_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╔
AssignVariableOp_3AssignVariableOp2assignvariableop_3_lstm_lstm_cell_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_4AssignVariableOp&assignvariableop_4_lstm_lstm_cell_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:│
AssignVariableOp_5AssignVariableOpassignvariableop_5_iterationIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_6AssignVariableOp assignvariableop_6_learning_rateIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_7AssignVariableOp/assignvariableop_7_adam_m_lstm_lstm_cell_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_8AssignVariableOp/assignvariableop_8_adam_v_lstm_lstm_cell_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_9AssignVariableOp9assignvariableop_9_adam_m_lstm_lstm_cell_recurrent_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_10AssignVariableOp:assignvariableop_10_adam_v_lstm_lstm_cell_recurrent_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_11AssignVariableOp.assignvariableop_11_adam_m_lstm_lstm_cell_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_12AssignVariableOp.assignvariableop_12_adam_v_lstm_lstm_cell_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_m_dense_7_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_v_dense_7_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_m_dense_7_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_v_dense_7_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ы
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: ║
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_20Identity_20:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%!

_user_specified_namecount:%!

_user_specified_nametotal:3/
-
_user_specified_nameAdam/v/dense_7/bias:3/
-
_user_specified_nameAdam/m/dense_7/bias:51
/
_user_specified_nameAdam/v/dense_7/kernel:51
/
_user_specified_nameAdam/m/dense_7/kernel::6
4
_user_specified_nameAdam/v/lstm/lstm_cell/bias::6
4
_user_specified_nameAdam/m/lstm/lstm_cell/bias:FB
@
_user_specified_name(&Adam/v/lstm/lstm_cell/recurrent_kernel:F
B
@
_user_specified_name(&Adam/m/lstm/lstm_cell/recurrent_kernel:<	8
6
_user_specified_nameAdam/v/lstm/lstm_cell/kernel:<8
6
_user_specified_nameAdam/m/lstm/lstm_cell/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:3/
-
_user_specified_namelstm/lstm_cell/bias:?;
9
_user_specified_name!lstm/lstm_cell/recurrent_kernel:51
/
_user_specified_namelstm/lstm_cell/kernel:,(
&
_user_specified_namedense_7/bias:.*
(
_user_specified_namedense_7/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
┴
ѓ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_208779

inputs

states
states_11
matmul_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ђW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ђV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ђO
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ђ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ђU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ђW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ђL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ђd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ђY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ђ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ђ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ђm
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ђ:         ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:PL
(
_output_shapes
:         ђ
 
_user_specified_namestates:PL
(
_output_shapes
:         ђ
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
В	
Ь
(__inference_model_5_layer_call_fn_209296
input_2
unknown:	ђ
	unknown_0:
ђђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_209266o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name209292:&"
 
_user_specified_name209290:&"
 
_user_specified_name209288:&"
 
_user_specified_name209286:&"
 
_user_specified_name209284:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
Я
х
%__inference_lstm_layer_call_fn_209399
inputs_0
unknown:	ђ
	unknown_0:
ђђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_208864p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name209395:&"
 
_user_specified_name209393:&"
 
_user_specified_name209391:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
╔
ё
E__inference_lstm_cell_layer_call_and_return_conditional_losses_210118

inputs
states_0
states_11
matmul_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ђW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ђV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ђO
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ђ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ђU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ђW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ђL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ђd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ђY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ђ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ђ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ђm
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ђ:         ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_1:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к	
├
while_cond_209166
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_209166___redundant_placeholder04
0while_while_cond_209166___redundant_placeholder14
0while_while_cond_209166___redundant_placeholder24
0while_while_cond_209166___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ђ:         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
░9
э
@__inference_lstm_layer_call_and_return_conditional_losses_208864

inputs#
lstm_cell_208780:	ђ$
lstm_cell_208782:
ђђ
lstm_cell_208784:	ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskж
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_208780lstm_cell_208782lstm_cell_208784*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_208779n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ћ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_208780lstm_cell_208782lstm_cell_208784*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ђ:         ђ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_208794*
condR
while_cond_208793*M
output_shapes<
:: : : : :         ђ:         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ђ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ђ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ђN
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name208784:&"
 
_user_specified_name208782:&"
 
_user_specified_name208780:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╚
│
%__inference_lstm_layer_call_fn_209421

inputs
unknown:	ђ
	unknown_0:
ђђ
	unknown_1:	ђ
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_209252p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name209417:&"
 
_user_specified_name209415:&"
 
_user_specified_name209413:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Щ	
ш
C__inference_dense_7_layer_call_and_return_conditional_losses_209091

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╚
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_209106

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
:         _
IdentityIdentityExpandDims:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ю:
и
while_body_208989
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђF
2while_lstm_cell_matmul_1_readvariableop_resource_0:
ђђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђD
0while_lstm_cell_matmul_1_readvariableop_resource:
ђђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђю
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:         ђЂ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ђo
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:         ђљ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђЁ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:         ђl
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђћ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ђw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ђБ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ђ:         ђ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
к	
├
while_cond_209915
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_209915___redundant_placeholder04
0while_while_cond_209915___redundant_placeholder14
0while_while_cond_209915___redundant_placeholder24
0while_while_cond_209915___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ђ:         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ж%
л
while_body_208647
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_208671_0:	ђ,
while_lstm_cell_208673_0:
ђђ'
while_lstm_cell_208675_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_208671:	ђ*
while_lstm_cell_208673:
ђђ%
while_lstm_cell_208675:	ђѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Д
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_208671_0while_lstm_cell_208673_0while_lstm_cell_208675_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_208632r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ђ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:00while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ј
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ђј
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         ђR

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_208671while_lstm_cell_208671_0"2
while_lstm_cell_208673while_lstm_cell_208673_0"2
while_lstm_cell_208675while_lstm_cell_208675_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ђ:         ђ: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:&
"
 
_user_specified_name208675:&	"
 
_user_specified_name208673:&"
 
_user_specified_name208671:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
УJ
Є
@__inference_lstm_layer_call_and_return_conditional_losses_209252

inputs;
(lstm_cell_matmul_readvariableop_resource:	ђ>
*lstm_cell_matmul_1_readvariableop_resource:
ђђ8
)lstm_cell_biasadd_readvariableop_resource:	ђ
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЅ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0љ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђј
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0і
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЃ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЄ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ї
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:         ђr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ђc
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*(
_output_shapes
:         ђ~
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:         ђ`
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђѓ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ђ:         ђ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_209167*
condR
while_cond_209166*M
output_shapes<
:: : : : :         ђ:         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ђ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ђ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ђЊ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
░9
э
@__inference_lstm_layer_call_and_return_conditional_losses_208717

inputs#
lstm_cell_208633:	ђ$
lstm_cell_208635:
ђђ
lstm_cell_208637:	ђ
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskж
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_208633lstm_cell_208635lstm_cell_208637*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_208632n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ћ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_208633lstm_cell_208635lstm_cell_208637*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ђ:         ђ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_208647*
condR
while_cond_208646*M
output_shapes<
:: : : : :         ђ:         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ђ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ђ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ђN
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:&"
 
_user_specified_name208637:&"
 
_user_specified_name208635:&"
 
_user_specified_name208633:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
ю:
и
while_body_209626
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	ђF
2while_lstm_cell_matmul_1_readvariableop_resource_0:
ђђ@
1while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	ђD
0while_lstm_cell_matmul_1_readvariableop_resource:
ђђ>
/while_lstm_cell_biasadd_readvariableop_resource:	ђѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0┤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђю
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђЋ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0ъ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:         ђЂ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ђo
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:         ђљ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђЁ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:         ђl
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђћ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ж
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ђw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ђБ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ђ:         ђ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╔
ё
E__inference_lstm_cell_layer_call_and_return_conditional_losses_210086

inputs
states_0
states_11
matmul_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ђW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ђV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ђO
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ђ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ђU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ђW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ђL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ђd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ђY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ђ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ђ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ђm
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ђ:         ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_1:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
─
З
*__inference_lstm_cell_layer_call_fn_210037

inputs
states_0
states_1
unknown:	ђ
	unknown_0:
ђђ
	unknown_1:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_208632p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ђ:         ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name210029:&"
 
_user_specified_name210027:&"
 
_user_specified_name210025:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_1:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ќ
┬
C__inference_model_5_layer_call_and_return_conditional_losses_209098
input_2
lstm_209075:	ђ
lstm_209077:
ђђ
lstm_209079:	ђ!
dense_7_209092:	ђ
dense_7_209094:
identityѕбdense_7/StatefulPartitionedCallбlstm/StatefulPartitionedCallй
lambda_2/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_208928І
lstm/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0lstm_209075lstm_209077lstm_209079*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_209074І
dense_7/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_7_209092dense_7_209094*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_209091w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         c
NoOpNoOp ^dense_7/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:&"
 
_user_specified_name209094:&"
 
_user_specified_name209092:&"
 
_user_specified_name209079:&"
 
_user_specified_name209077:&"
 
_user_specified_name209075:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
┌
К
model_5_lstm_while_cond_2084786
2model_5_lstm_while_model_5_lstm_while_loop_counter<
8model_5_lstm_while_model_5_lstm_while_maximum_iterations"
model_5_lstm_while_placeholder$
 model_5_lstm_while_placeholder_1$
 model_5_lstm_while_placeholder_2$
 model_5_lstm_while_placeholder_38
4model_5_lstm_while_less_model_5_lstm_strided_slice_1N
Jmodel_5_lstm_while_model_5_lstm_while_cond_208478___redundant_placeholder0N
Jmodel_5_lstm_while_model_5_lstm_while_cond_208478___redundant_placeholder1N
Jmodel_5_lstm_while_model_5_lstm_while_cond_208478___redundant_placeholder2N
Jmodel_5_lstm_while_model_5_lstm_while_cond_208478___redundant_placeholder3
model_5_lstm_while_identity
ќ
model_5/lstm/while/LessLessmodel_5_lstm_while_placeholder4model_5_lstm_while_less_model_5_lstm_strided_slice_1*
T0*
_output_shapes
: e
model_5/lstm/while/IdentityIdentitymodel_5/lstm/while/Less:z:0*
T0
*
_output_shapes
: "C
model_5_lstm_while_identity$model_5/lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ђ:         ђ: :::::

_output_shapes
::TP

_output_shapes
: 
6
_user_specified_namemodel_5/lstm/strided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :]Y

_output_shapes
: 
?
_user_specified_name'%model_5/lstm/while/maximum_iterations:W S

_output_shapes
: 
9
_user_specified_name!model_5/lstm/while/loop_counter
ћf
Ќ
!__inference__wrapped_model_208570
input_2H
5model_5_lstm_lstm_cell_matmul_readvariableop_resource:	ђK
7model_5_lstm_lstm_cell_matmul_1_readvariableop_resource:
ђђE
6model_5_lstm_lstm_cell_biasadd_readvariableop_resource:	ђA
.model_5_dense_7_matmul_readvariableop_resource:	ђ=
/model_5_dense_7_biasadd_readvariableop_resource:
identityѕб&model_5/dense_7/BiasAdd/ReadVariableOpб%model_5/dense_7/MatMul/ReadVariableOpб-model_5/lstm/lstm_cell/BiasAdd/ReadVariableOpб,model_5/lstm/lstm_cell/MatMul/ReadVariableOpб.model_5/lstm/lstm_cell/MatMul_1/ReadVariableOpбmodel_5/lstm/whilea
model_5/lambda_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :њ
model_5/lambda_2/ExpandDims
ExpandDimsinput_2(model_5/lambda_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         t
model_5/lstm/ShapeShape$model_5/lambda_2/ExpandDims:output:0*
T0*
_output_shapes
::ь¤j
 model_5/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"model_5/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"model_5/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
model_5/lstm/strided_sliceStridedSlicemodel_5/lstm/Shape:output:0)model_5/lstm/strided_slice/stack:output:0+model_5/lstm/strided_slice/stack_1:output:0+model_5/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
model_5/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђџ
model_5/lstm/zeros/packedPack#model_5/lstm/strided_slice:output:0$model_5/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
model_5/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ћ
model_5/lstm/zerosFill"model_5/lstm/zeros/packed:output:0!model_5/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:         ђ`
model_5/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђъ
model_5/lstm/zeros_1/packedPack#model_5/lstm/strided_slice:output:0&model_5/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:_
model_5/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    џ
model_5/lstm/zeros_1Fill$model_5/lstm/zeros_1/packed:output:0#model_5/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ђp
model_5/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
model_5/lstm/transpose	Transpose$model_5/lambda_2/ExpandDims:output:0$model_5/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:         l
model_5/lstm/Shape_1Shapemodel_5/lstm/transpose:y:0*
T0*
_output_shapes
::ь¤l
"model_5/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model_5/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model_5/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
model_5/lstm/strided_slice_1StridedSlicemodel_5/lstm/Shape_1:output:0+model_5/lstm/strided_slice_1/stack:output:0-model_5/lstm/strided_slice_1/stack_1:output:0-model_5/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(model_5/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         █
model_5/lstm/TensorArrayV2TensorListReserve1model_5/lstm/TensorArrayV2/element_shape:output:0%model_5/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЊ
Bmodel_5/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Є
4model_5/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_5/lstm/transpose:y:0Kmodel_5/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмl
"model_5/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model_5/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model_5/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
model_5/lstm/strided_slice_2StridedSlicemodel_5/lstm/transpose:y:0+model_5/lstm/strided_slice_2/stack:output:0-model_5/lstm/strided_slice_2/stack_1:output:0-model_5/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskБ
,model_5/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp5model_5_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0и
model_5/lstm/lstm_cell/MatMulMatMul%model_5/lstm/strided_slice_2:output:04model_5/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђе
.model_5/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7model_5_lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0▒
model_5/lstm/lstm_cell/MatMul_1MatMulmodel_5/lstm/zeros:output:06model_5/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђф
model_5/lstm/lstm_cell/addAddV2'model_5/lstm/lstm_cell/MatMul:product:0)model_5/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђА
-model_5/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6model_5_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0│
model_5/lstm/lstm_cell/BiasAddBiasAddmodel_5/lstm/lstm_cell/add:z:05model_5/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђh
&model_5/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
model_5/lstm/lstm_cell/splitSplit/model_5/lstm/lstm_cell/split/split_dim:output:0'model_5/lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitЃ
model_5/lstm/lstm_cell/SigmoidSigmoid%model_5/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:         ђЁ
 model_5/lstm/lstm_cell/Sigmoid_1Sigmoid%model_5/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:         ђЎ
model_5/lstm/lstm_cell/mulMul$model_5/lstm/lstm_cell/Sigmoid_1:y:0model_5/lstm/zeros_1:output:0*
T0*(
_output_shapes
:         ђ}
model_5/lstm/lstm_cell/ReluRelu%model_5/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:         ђЦ
model_5/lstm/lstm_cell/mul_1Mul"model_5/lstm/lstm_cell/Sigmoid:y:0)model_5/lstm/lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђџ
model_5/lstm/lstm_cell/add_1AddV2model_5/lstm/lstm_cell/mul:z:0 model_5/lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђЁ
 model_5/lstm/lstm_cell/Sigmoid_2Sigmoid%model_5/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:         ђz
model_5/lstm/lstm_cell/Relu_1Relu model_5/lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђЕ
model_5/lstm/lstm_cell/mul_2Mul$model_5/lstm/lstm_cell/Sigmoid_2:y:0+model_5/lstm/lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђ{
*model_5/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   k
)model_5/lstm/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :В
model_5/lstm/TensorArrayV2_1TensorListReserve3model_5/lstm/TensorArrayV2_1/element_shape:output:02model_5/lstm/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмS
model_5/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%model_5/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         a
model_5/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ќ
model_5/lstm/whileWhile(model_5/lstm/while/loop_counter:output:0.model_5/lstm/while/maximum_iterations:output:0model_5/lstm/time:output:0%model_5/lstm/TensorArrayV2_1:handle:0model_5/lstm/zeros:output:0model_5/lstm/zeros_1:output:0%model_5/lstm/strided_slice_1:output:0Dmodel_5/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:05model_5_lstm_lstm_cell_matmul_readvariableop_resource7model_5_lstm_lstm_cell_matmul_1_readvariableop_resource6model_5_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ђ:         ђ: : : : : *%
_read_only_resource_inputs
	
**
body"R 
model_5_lstm_while_body_208479**
cond"R 
model_5_lstm_while_cond_208478*M
output_shapes<
:: : : : :         ђ:         ђ: : : : : *
parallel_iterations ј
=model_5/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ■
/model_5/lstm/TensorArrayV2Stack/TensorListStackTensorListStackmodel_5/lstm/while:output:3Fmodel_5/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ђ*
element_dtype0*
num_elementsu
"model_5/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         n
$model_5/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$model_5/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
model_5/lstm/strided_slice_3StridedSlice8model_5/lstm/TensorArrayV2Stack/TensorListStack:tensor:0+model_5/lstm/strided_slice_3/stack:output:0-model_5/lstm/strided_slice_3/stack_1:output:0-model_5/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maskr
model_5/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Й
model_5/lstm/transpose_1	Transpose8model_5/lstm/TensorArrayV2Stack/TensorListStack:tensor:0&model_5/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ђh
model_5/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ћ
%model_5/dense_7/MatMul/ReadVariableOpReadVariableOp.model_5_dense_7_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0е
model_5/dense_7/MatMulMatMul%model_5/lstm/strided_slice_3:output:0-model_5/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         њ
&model_5/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_5_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0д
model_5/dense_7/BiasAddBiasAdd model_5/dense_7/MatMul:product:0.model_5/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         o
IdentityIdentity model_5/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ў
NoOpNoOp'^model_5/dense_7/BiasAdd/ReadVariableOp&^model_5/dense_7/MatMul/ReadVariableOp.^model_5/lstm/lstm_cell/BiasAdd/ReadVariableOp-^model_5/lstm/lstm_cell/MatMul/ReadVariableOp/^model_5/lstm/lstm_cell/MatMul_1/ReadVariableOp^model_5/lstm/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : : : 2P
&model_5/dense_7/BiasAdd/ReadVariableOp&model_5/dense_7/BiasAdd/ReadVariableOp2N
%model_5/dense_7/MatMul/ReadVariableOp%model_5/dense_7/MatMul/ReadVariableOp2^
-model_5/lstm/lstm_cell/BiasAdd/ReadVariableOp-model_5/lstm/lstm_cell/BiasAdd/ReadVariableOp2\
,model_5/lstm/lstm_cell/MatMul/ReadVariableOp,model_5/lstm/lstm_cell/MatMul/ReadVariableOp2`
.model_5/lstm/lstm_cell/MatMul_1/ReadVariableOp.model_5/lstm/lstm_cell/MatMul_1/ReadVariableOp2(
model_5/lstm/whilemodel_5/lstm/while:($
"
_user_specified_name
resource:($
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
resource:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
╚
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_209377

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
:         _
IdentityIdentityExpandDims:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╚
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_209371

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
:         _
IdentityIdentityExpandDims:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ѕJ
О
model_5_lstm_while_body_2084796
2model_5_lstm_while_model_5_lstm_while_loop_counter<
8model_5_lstm_while_model_5_lstm_while_maximum_iterations"
model_5_lstm_while_placeholder$
 model_5_lstm_while_placeholder_1$
 model_5_lstm_while_placeholder_2$
 model_5_lstm_while_placeholder_35
1model_5_lstm_while_model_5_lstm_strided_slice_1_0q
mmodel_5_lstm_while_tensorarrayv2read_tensorlistgetitem_model_5_lstm_tensorarrayunstack_tensorlistfromtensor_0P
=model_5_lstm_while_lstm_cell_matmul_readvariableop_resource_0:	ђS
?model_5_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:
ђђM
>model_5_lstm_while_lstm_cell_biasadd_readvariableop_resource_0:	ђ
model_5_lstm_while_identity!
model_5_lstm_while_identity_1!
model_5_lstm_while_identity_2!
model_5_lstm_while_identity_3!
model_5_lstm_while_identity_4!
model_5_lstm_while_identity_53
/model_5_lstm_while_model_5_lstm_strided_slice_1o
kmodel_5_lstm_while_tensorarrayv2read_tensorlistgetitem_model_5_lstm_tensorarrayunstack_tensorlistfromtensorN
;model_5_lstm_while_lstm_cell_matmul_readvariableop_resource:	ђQ
=model_5_lstm_while_lstm_cell_matmul_1_readvariableop_resource:
ђђK
<model_5_lstm_while_lstm_cell_biasadd_readvariableop_resource:	ђѕб3model_5/lstm/while/lstm_cell/BiasAdd/ReadVariableOpб2model_5/lstm/while/lstm_cell/MatMul/ReadVariableOpб4model_5/lstm/while/lstm_cell/MatMul_1/ReadVariableOpЋ
Dmodel_5/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       у
6model_5/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmmodel_5_lstm_while_tensorarrayv2read_tensorlistgetitem_model_5_lstm_tensorarrayunstack_tensorlistfromtensor_0model_5_lstm_while_placeholderMmodel_5/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0▒
2model_5/lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp=model_5_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0█
#model_5/lstm/while/lstm_cell/MatMulMatMul=model_5/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0:model_5/lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђХ
4model_5/lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp?model_5_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0┬
%model_5/lstm/while/lstm_cell/MatMul_1MatMul model_5_lstm_while_placeholder_2<model_5/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ╝
 model_5/lstm/while/lstm_cell/addAddV2-model_5/lstm/while/lstm_cell/MatMul:product:0/model_5/lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ»
3model_5/lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp>model_5_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0┼
$model_5/lstm/while/lstm_cell/BiasAddBiasAdd$model_5/lstm/while/lstm_cell/add:z:0;model_5/lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђn
,model_5/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
"model_5/lstm/while/lstm_cell/splitSplit5model_5/lstm/while/lstm_cell/split/split_dim:output:0-model_5/lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ђ:         ђ:         ђ:         ђ*
	num_splitЈ
$model_5/lstm/while/lstm_cell/SigmoidSigmoid+model_5/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:         ђЉ
&model_5/lstm/while/lstm_cell/Sigmoid_1Sigmoid+model_5/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:         ђе
 model_5/lstm/while/lstm_cell/mulMul*model_5/lstm/while/lstm_cell/Sigmoid_1:y:0 model_5_lstm_while_placeholder_3*
T0*(
_output_shapes
:         ђЅ
!model_5/lstm/while/lstm_cell/ReluRelu+model_5/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:         ђи
"model_5/lstm/while/lstm_cell/mul_1Mul(model_5/lstm/while/lstm_cell/Sigmoid:y:0/model_5/lstm/while/lstm_cell/Relu:activations:0*
T0*(
_output_shapes
:         ђг
"model_5/lstm/while/lstm_cell/add_1AddV2$model_5/lstm/while/lstm_cell/mul:z:0&model_5/lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:         ђЉ
&model_5/lstm/while/lstm_cell/Sigmoid_2Sigmoid+model_5/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:         ђє
#model_5/lstm/while/lstm_cell/Relu_1Relu&model_5/lstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:         ђ╗
"model_5/lstm/while/lstm_cell/mul_2Mul*model_5/lstm/while/lstm_cell/Sigmoid_2:y:01model_5/lstm/while/lstm_cell/Relu_1:activations:0*
T0*(
_output_shapes
:         ђ
=model_5/lstm/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ъ
7model_5/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem model_5_lstm_while_placeholder_1Fmodel_5/lstm/while/TensorArrayV2Write/TensorListSetItem/index:output:0&model_5/lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмZ
model_5/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
model_5/lstm/while/addAddV2model_5_lstm_while_placeholder!model_5/lstm/while/add/y:output:0*
T0*
_output_shapes
: \
model_5/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Џ
model_5/lstm/while/add_1AddV22model_5_lstm_while_model_5_lstm_while_loop_counter#model_5/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: ђ
model_5/lstm/while/IdentityIdentitymodel_5/lstm/while/add_1:z:0^model_5/lstm/while/NoOp*
T0*
_output_shapes
: ъ
model_5/lstm/while/Identity_1Identity8model_5_lstm_while_model_5_lstm_while_maximum_iterations^model_5/lstm/while/NoOp*
T0*
_output_shapes
: ђ
model_5/lstm/while/Identity_2Identitymodel_5/lstm/while/add:z:0^model_5/lstm/while/NoOp*
T0*
_output_shapes
: Г
model_5/lstm/while/Identity_3IdentityGmodel_5/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model_5/lstm/while/NoOp*
T0*
_output_shapes
: ъ
model_5/lstm/while/Identity_4Identity&model_5/lstm/while/lstm_cell/mul_2:z:0^model_5/lstm/while/NoOp*
T0*(
_output_shapes
:         ђъ
model_5/lstm/while/Identity_5Identity&model_5/lstm/while/lstm_cell/add_1:z:0^model_5/lstm/while/NoOp*
T0*(
_output_shapes
:         ђО
model_5/lstm/while/NoOpNoOp4^model_5/lstm/while/lstm_cell/BiasAdd/ReadVariableOp3^model_5/lstm/while/lstm_cell/MatMul/ReadVariableOp5^model_5/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "G
model_5_lstm_while_identity_1&model_5/lstm/while/Identity_1:output:0"G
model_5_lstm_while_identity_2&model_5/lstm/while/Identity_2:output:0"G
model_5_lstm_while_identity_3&model_5/lstm/while/Identity_3:output:0"G
model_5_lstm_while_identity_4&model_5/lstm/while/Identity_4:output:0"G
model_5_lstm_while_identity_5&model_5/lstm/while/Identity_5:output:0"C
model_5_lstm_while_identity$model_5/lstm/while/Identity:output:0"~
<model_5_lstm_while_lstm_cell_biasadd_readvariableop_resource>model_5_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"ђ
=model_5_lstm_while_lstm_cell_matmul_1_readvariableop_resource?model_5_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"|
;model_5_lstm_while_lstm_cell_matmul_readvariableop_resource=model_5_lstm_while_lstm_cell_matmul_readvariableop_resource_0"d
/model_5_lstm_while_model_5_lstm_strided_slice_11model_5_lstm_while_model_5_lstm_strided_slice_1_0"▄
kmodel_5_lstm_while_tensorarrayv2read_tensorlistgetitem_model_5_lstm_tensorarrayunstack_tensorlistfromtensormmodel_5_lstm_while_tensorarrayv2read_tensorlistgetitem_model_5_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ђ:         ђ: : : : : 2j
3model_5/lstm/while/lstm_cell/BiasAdd/ReadVariableOp3model_5/lstm/while/lstm_cell/BiasAdd/ReadVariableOp2h
2model_5/lstm/while/lstm_cell/MatMul/ReadVariableOp2model_5/lstm/while/lstm_cell/MatMul/ReadVariableOp2l
4model_5/lstm/while/lstm_cell/MatMul_1/ReadVariableOp4model_5/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:lh

_output_shapes
: 
N
_user_specified_name64model_5/lstm/TensorArrayUnstack/TensorListFromTensor:TP

_output_shapes
: 
6
_user_specified_namemodel_5/lstm/strided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :]Y

_output_shapes
: 
?
_user_specified_name'%model_5/lstm/while/maximum_iterations:W S

_output_shapes
: 
9
_user_specified_name!model_5/lstm/while/loop_counter
к	
├
while_cond_208646
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_208646___redundant_placeholder04
0while_while_cond_208646___redundant_placeholder14
0while_while_cond_208646___redundant_placeholder24
0while_while_cond_208646___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ђ:         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter"╩L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ф
serving_defaultќ
;
input_20
serving_default_input_2:0         ;
dense_70
StatefulPartitionedCall:0         tensorflow/serving/predict:БА
╦
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
┌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
╗
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
C
%0
&1
'2
#3
$4"
trackable_list_wrapper
C
%0
&1
'2
#3
$4"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
├
-trace_0
.trace_12ї
(__inference_model_5_layer_call_fn_209281
(__inference_model_5_layer_call_fn_209296х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z-trace_0z.trace_1
щ
/trace_0
0trace_12┬
C__inference_model_5_layer_call_and_return_conditional_losses_209098
C__inference_model_5_layer_call_and_return_conditional_losses_209266х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z/trace_0z0trace_1
╠B╔
!__inference__wrapped_model_208570input_2"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ю
1
_variables
2_iterations
3_learning_rate
4_index_dict
5
_momentums
6_velocities
7_update_step_xla"
experimentalOptimizer
,
8serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
┼
>trace_0
?trace_12ј
)__inference_lambda_2_layer_call_fn_209360
)__inference_lambda_2_layer_call_fn_209365х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z>trace_0z?trace_1
ч
@trace_0
Atrace_12─
D__inference_lambda_2_layer_call_and_return_conditional_losses_209371
D__inference_lambda_2_layer_call_and_return_conditional_losses_209377х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z@trace_0zAtrace_1
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

Bstates
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
н
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32ж
%__inference_lstm_layer_call_fn_209388
%__inference_lstm_layer_call_fn_209399
%__inference_lstm_layer_call_fn_209410
%__inference_lstm_layer_call_fn_209421╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zHtrace_0zItrace_1zJtrace_2zKtrace_3
└
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32Н
@__inference_lstm_layer_call_and_return_conditional_losses_209566
@__inference_lstm_layer_call_and_return_conditional_losses_209711
@__inference_lstm_layer_call_and_return_conditional_losses_209856
@__inference_lstm_layer_call_and_return_conditional_losses_210001╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
"
_generic_user_object
Э
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
V_random_generator
W
state_size

%kernel
&recurrent_kernel
'bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Р
]trace_02┼
(__inference_dense_7_layer_call_fn_210010ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z]trace_0
§
^trace_02Я
C__inference_dense_7_layer_call_and_return_conditional_losses_210020ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z^trace_0
!:	ђ2dense_7/kernel
:2dense_7/bias
(:&	ђ2lstm/lstm_cell/kernel
3:1
ђђ2lstm/lstm_cell/recurrent_kernel
": ђ2lstm/lstm_cell/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
_0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
­Bь
(__inference_model_5_layer_call_fn_209281input_2"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­Bь
(__inference_model_5_layer_call_fn_209296input_2"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ІBѕ
C__inference_model_5_layer_call_and_return_conditional_losses_209098input_2"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ІBѕ
C__inference_model_5_layer_call_and_return_conditional_losses_209266input_2"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
n
20
`1
a2
b3
c4
d5
e6
f7
g8
h9
i10"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
C
`0
b1
d2
f3
h4"
trackable_list_wrapper
C
a0
c1
e2
g3
i4"
trackable_list_wrapper
х2▓»
д▓б
FullArgSpec*
args"џ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
╦B╚
$__inference_signature_wrapper_209355input_2"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
­Bь
)__inference_lambda_2_layer_call_fn_209360inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­Bь
)__inference_lambda_2_layer_call_fn_209365inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ІBѕ
D__inference_lambda_2_layer_call_and_return_conditional_losses_209371inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ІBѕ
D__inference_lambda_2_layer_call_and_return_conditional_losses_209377inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЃBђ
%__inference_lstm_layer_call_fn_209388inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЃBђ
%__inference_lstm_layer_call_fn_209399inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
%__inference_lstm_layer_call_fn_209410inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
%__inference_lstm_layer_call_fn_209421inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ъBЏ
@__inference_lstm_layer_call_and_return_conditional_losses_209566inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ъBЏ
@__inference_lstm_layer_call_and_return_conditional_losses_209711inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
@__inference_lstm_layer_call_and_return_conditional_losses_209856inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
@__inference_lstm_layer_call_and_return_conditional_losses_210001inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
┼
otrace_0
ptrace_12ј
*__inference_lstm_cell_layer_call_fn_210037
*__inference_lstm_cell_layer_call_fn_210054│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zotrace_0zptrace_1
ч
qtrace_0
rtrace_12─
E__inference_lstm_cell_layer_call_and_return_conditional_losses_210086
E__inference_lstm_cell_layer_call_and_return_conditional_losses_210118│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zqtrace_0zrtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
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
мB¤
(__inference_dense_7_layer_call_fn_210010inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ьBЖ
C__inference_dense_7_layer_call_and_return_conditional_losses_210020inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
N
s	variables
t	keras_api
	utotal
	vcount"
_tf_keras_metric
-:+	ђ2Adam/m/lstm/lstm_cell/kernel
-:+	ђ2Adam/v/lstm/lstm_cell/kernel
8:6
ђђ2&Adam/m/lstm/lstm_cell/recurrent_kernel
8:6
ђђ2&Adam/v/lstm/lstm_cell/recurrent_kernel
':%ђ2Adam/m/lstm/lstm_cell/bias
':%ђ2Adam/v/lstm/lstm_cell/bias
&:$	ђ2Adam/m/dense_7/kernel
&:$	ђ2Adam/v/dense_7/kernel
:2Adam/m/dense_7/bias
:2Adam/v/dense_7/bias
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
ЃBђ
*__inference_lstm_cell_layer_call_fn_210037inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЃBђ
*__inference_lstm_cell_layer_call_fn_210054inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ъBЏ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_210086inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ъBЏ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_210118inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
u0
v1"
trackable_list_wrapper
-
s	variables"
_generic_user_object
:  (2total
:  (2countЉ
!__inference__wrapped_model_208570l%&'#$0б-
&б#
!і
input_2         
ф "1ф.
,
dense_7!і
dense_7         Ф
C__inference_dense_7_layer_call_and_return_conditional_losses_210020d#$0б-
&б#
!і
inputs         ђ
ф ",б)
"і
tensor_0         
џ Ё
(__inference_dense_7_layer_call_fn_210010Y#$0б-
&б#
!і
inputs         ђ
ф "!і
unknown         │
D__inference_lambda_2_layer_call_and_return_conditional_losses_209371k7б4
-б*
 і
inputs         

 
p
ф "0б-
&і#
tensor_0         
џ │
D__inference_lambda_2_layer_call_and_return_conditional_losses_209377k7б4
-б*
 і
inputs         

 
p 
ф "0б-
&і#
tensor_0         
џ Ї
)__inference_lambda_2_layer_call_fn_209360`7б4
-б*
 і
inputs         

 
p
ф "%і"
unknown         Ї
)__inference_lambda_2_layer_call_fn_209365`7б4
-б*
 і
inputs         

 
p 
ф "%і"
unknown         С
E__inference_lstm_cell_layer_call_and_return_conditional_losses_210086џ%&'ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ђ
#і 
states_1         ђ
p
ф "ЇбЅ
Ђб~
%і"

tensor_0_0         ђ
UџR
'і$
tensor_0_1_0         ђ
'і$
tensor_0_1_1         ђ
џ С
E__inference_lstm_cell_layer_call_and_return_conditional_losses_210118џ%&'ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ђ
#і 
states_1         ђ
p 
ф "ЇбЅ
Ђб~
%і"

tensor_0_0         ђ
UџR
'і$
tensor_0_1_0         ђ
'і$
tensor_0_1_1         ђ
џ Х
*__inference_lstm_cell_layer_call_fn_210037Є%&'ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ђ
#і 
states_1         ђ
p
ф "{бx
#і 
tensor_0         ђ
QџN
%і"

tensor_1_0         ђ
%і"

tensor_1_1         ђХ
*__inference_lstm_cell_layer_call_fn_210054Є%&'ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ђ
#і 
states_1         ђ
p 
ф "{бx
#і 
tensor_0         ђ
QџN
%і"

tensor_1_0         ђ
%і"

tensor_1_1         ђ╩
@__inference_lstm_layer_call_and_return_conditional_losses_209566Ё%&'OбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "-б*
#і 
tensor_0         ђ
џ ╩
@__inference_lstm_layer_call_and_return_conditional_losses_209711Ё%&'OбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "-б*
#і 
tensor_0         ђ
џ ╣
@__inference_lstm_layer_call_and_return_conditional_losses_209856u%&'?б<
5б2
$і!
inputs         

 
p

 
ф "-б*
#і 
tensor_0         ђ
џ ╣
@__inference_lstm_layer_call_and_return_conditional_losses_210001u%&'?б<
5б2
$і!
inputs         

 
p 

 
ф "-б*
#і 
tensor_0         ђ
џ Б
%__inference_lstm_layer_call_fn_209388z%&'OбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф ""і
unknown         ђБ
%__inference_lstm_layer_call_fn_209399z%&'OбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф ""і
unknown         ђЊ
%__inference_lstm_layer_call_fn_209410j%&'?б<
5б2
$і!
inputs         

 
p

 
ф ""і
unknown         ђЊ
%__inference_lstm_layer_call_fn_209421j%&'?б<
5б2
$і!
inputs         

 
p 

 
ф ""і
unknown         ђХ
C__inference_model_5_layer_call_and_return_conditional_losses_209098o%&'#$8б5
.б+
!і
input_2         
p

 
ф ",б)
"і
tensor_0         
џ Х
C__inference_model_5_layer_call_and_return_conditional_losses_209266o%&'#$8б5
.б+
!і
input_2         
p 

 
ф ",б)
"і
tensor_0         
џ љ
(__inference_model_5_layer_call_fn_209281d%&'#$8б5
.б+
!і
input_2         
p

 
ф "!і
unknown         љ
(__inference_model_5_layer_call_fn_209296d%&'#$8б5
.б+
!і
input_2         
p 

 
ф "!і
unknown         Ъ
$__inference_signature_wrapper_209355w%&'#$;б8
б 
1ф.
,
input_2!і
input_2         "1ф.
,
dense_7!і
dense_7         
??
??
?
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
?
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
?
&QNetwork/EncodingNetwork/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&QNetwork/EncodingNetwork/conv2d/kernel
?
:QNetwork/EncodingNetwork/conv2d/kernel/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/conv2d/kernel*&
_output_shapes
: *
dtype0
?
$QNetwork/EncodingNetwork/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$QNetwork/EncodingNetwork/conv2d/bias
?
8QNetwork/EncodingNetwork/conv2d/bias/Read/ReadVariableOpReadVariableOp$QNetwork/EncodingNetwork/conv2d/bias*
_output_shapes
: *
dtype0
?
(QNetwork/EncodingNetwork/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *9
shared_name*(QNetwork/EncodingNetwork/conv2d_1/kernel
?
<QNetwork/EncodingNetwork/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/conv2d_1/kernel*&
_output_shapes
:  *
dtype0
?
&QNetwork/EncodingNetwork/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&QNetwork/EncodingNetwork/conv2d_1/bias
?
:QNetwork/EncodingNetwork/conv2d_1/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/conv2d_1/bias*
_output_shapes
: *
dtype0
?
%QNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *6
shared_name'%QNetwork/EncodingNetwork/dense/kernel
?
9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	? *
dtype0
?
#QNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#QNetwork/EncodingNetwork/dense/bias
?
7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp#QNetwork/EncodingNetwork/dense/bias*
_output_shapes
: *
dtype0
?
'QNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*8
shared_name)'QNetwork/EncodingNetwork/dense_1/kernel
?
;QNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp'QNetwork/EncodingNetwork/dense_1/kernel*
_output_shapes

: @*
dtype0
?
%QNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%QNetwork/EncodingNetwork/dense_1/bias
?
9QNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense_1/bias*
_output_shapes
:@*
dtype0
?
'QNetwork/EncodingNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*8
shared_name)'QNetwork/EncodingNetwork/dense_2/kernel
?
;QNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOp'QNetwork/EncodingNetwork/dense_2/kernel*
_output_shapes
:	@?*
dtype0
?
%QNetwork/EncodingNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%QNetwork/EncodingNetwork/dense_2/bias
?
9QNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense_2/bias*
_output_shapes	
:?*
dtype0
?
QNetwork/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameQNetwork/dense_3/kernel
?
+QNetwork/dense_3/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_3/kernel*
_output_shapes
:	?*
dtype0
?
QNetwork/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameQNetwork/dense_3/bias
{
)QNetwork/dense_3/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?%B?% B?%
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
V
0
1
2
	3

4
5
6
7
8
9
10
11

0
 
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/conv2d/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE$QNetwork/EncodingNetwork/conv2d/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/conv2d_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/conv2d_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE#QNetwork/EncodingNetwork/dense/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE'QNetwork/EncodingNetwork/dense_1/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense_1/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE'QNetwork/EncodingNetwork/dense_2/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense_2/bias,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEQNetwork/dense_3/kernel-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEQNetwork/dense_3/bias-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE

ref
1


_q_network
t
_encoder
_q_value_layer
	variables
regularization_losses
trainable_variables
	keras_api
n
_postprocessing_layers
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
V
0
1
2
	3

4
5
6
7
8
9
10
11
 
V
0
1
2
	3

4
5
6
7
8
9
10
11
?
	variables
regularization_losses
$non_trainable_variables

%layers
&layer_metrics
'metrics
(layer_regularization_losses
trainable_variables
*
)0
*1
+2
,3
-4
.5
F
0
1
2
	3

4
5
6
7
8
9
 
F
0
1
2
	3

4
5
6
7
8
9
?
	variables
regularization_losses
/non_trainable_variables

0layers
1layer_metrics
2metrics
3layer_regularization_losses
trainable_variables

0
1
 

0
1
?
 	variables
!regularization_losses
4non_trainable_variables

5layers
6layer_metrics
7metrics
8layer_regularization_losses
"trainable_variables
 

0
1
 
 
 
h

kernel
bias
9	variables
:regularization_losses
;trainable_variables
<	keras_api
h

kernel
	bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
R
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
h


kernel
bias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
h

kernel
bias
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
h

kernel
bias
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
 
*
)0
*1
+2
,3
-4
.5
 
 
 
 
 
 
 
 

0
1
 

0
1
?
9	variables
:regularization_losses
Qnon_trainable_variables

Rlayers
Slayer_metrics
Tmetrics
Ulayer_regularization_losses
;trainable_variables

0
	1
 

0
	1
?
=	variables
>regularization_losses
Vnon_trainable_variables

Wlayers
Xlayer_metrics
Ymetrics
Zlayer_regularization_losses
?trainable_variables
 
 
 
?
A	variables
Bregularization_losses
[non_trainable_variables

\layers
]layer_metrics
^metrics
_layer_regularization_losses
Ctrainable_variables


0
1
 


0
1
?
E	variables
Fregularization_losses
`non_trainable_variables

alayers
blayer_metrics
cmetrics
dlayer_regularization_losses
Gtrainable_variables

0
1
 

0
1
?
I	variables
Jregularization_losses
enon_trainable_variables

flayers
glayer_metrics
hmetrics
ilayer_regularization_losses
Ktrainable_variables

0
1
 

0
1
?
M	variables
Nregularization_losses
jnon_trainable_variables

klayers
llayer_metrics
mmetrics
nlayer_regularization_losses
Otrainable_variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
action_0/observationPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
j
action_0/rewardPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
m
action_0/step_typePlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type&QNetwork/EncodingNetwork/conv2d/kernel$QNetwork/EncodingNetwork/conv2d/bias(QNetwork/EncodingNetwork/conv2d_1/kernel&QNetwork/EncodingNetwork/conv2d_1/bias%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/bias'QNetwork/EncodingNetwork/dense_1/kernel%QNetwork/EncodingNetwork/dense_1/bias'QNetwork/EncodingNetwork/dense_2/kernel%QNetwork/EncodingNetwork/dense_2/biasQNetwork/dense_3/kernelQNetwork/dense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_14824940
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_14824952
?
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_14824974
?
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_14824967
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp:QNetwork/EncodingNetwork/conv2d/kernel/Read/ReadVariableOp8QNetwork/EncodingNetwork/conv2d/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/conv2d_1/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/conv2d_1/bias/Read/ReadVariableOp9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOp7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOp;QNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOp9QNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOp;QNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOp9QNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOp+QNetwork/dense_3/kernel/Read/ReadVariableOp)QNetwork/dense_3/bias/Read/ReadVariableOpConst*
Tin
2	*
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
GPU 2J 8? **
f%R#
!__inference__traced_save_14825256
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable&QNetwork/EncodingNetwork/conv2d/kernel$QNetwork/EncodingNetwork/conv2d/bias(QNetwork/EncodingNetwork/conv2d_1/kernel&QNetwork/EncodingNetwork/conv2d_1/bias%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/bias'QNetwork/EncodingNetwork/dense_1/kernel%QNetwork/EncodingNetwork/dense_1/bias'QNetwork/EncodingNetwork/dense_2/kernel%QNetwork/EncodingNetwork/dense_2/biasQNetwork/dense_3/kernelQNetwork/dense_3/bias*
Tin
2*
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_14825305??
?|
?
*__inference_polymorphic_action_fn_14824879
	time_step
time_step_1
time_step_2
time_step_3B
>qnetwork_encodingnetwork_conv2d_conv2d_readvariableop_resourceC
?qnetwork_encodingnetwork_conv2d_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_conv2d_1_conv2d_readvariableop_resourceE
Aqnetwork_encodingnetwork_conv2d_1_biasadd_readvariableop_resourceA
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceB
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceC
?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resourceC
?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource3
/qnetwork_dense_3_matmul_readvariableop_resource4
0qnetwork_dense_3_biasadd_readvariableop_resource
identity??6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp?5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp?5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?'QNetwork/dense_3/BiasAdd/ReadVariableOp?&QNetwork/dense_3/MatMul/ReadVariableOp?
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp?
&QNetwork/EncodingNetwork/conv2d/Conv2DConv2Dtime_step_3=QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2(
&QNetwork/EncodingNetwork/conv2d/Conv2D?
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp?
'QNetwork/EncodingNetwork/conv2d/BiasAddBiasAdd/QNetwork/EncodingNetwork/conv2d/Conv2D:output:0>QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2)
'QNetwork/EncodingNetwork/conv2d/BiasAdd?
$QNetwork/EncodingNetwork/conv2d/ReluRelu0QNetwork/EncodingNetwork/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2&
$QNetwork/EncodingNetwork/conv2d/Relu?
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype029
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp?
(QNetwork/EncodingNetwork/conv2d_1/Conv2DConv2D2QNetwork/EncodingNetwork/conv2d/Relu:activations:0?QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2*
(QNetwork/EncodingNetwork/conv2d_1/Conv2D?
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/conv2d_1/Conv2D:output:0@QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2+
)QNetwork/EncodingNetwork/conv2d_1/BiasAdd?
&QNetwork/EncodingNetwork/conv2d_1/ReluRelu2QNetwork/EncodingNetwork/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2(
&QNetwork/EncodingNetwork/conv2d_1/Relu?
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&QNetwork/EncodingNetwork/flatten/Const?
(QNetwork/EncodingNetwork/flatten/ReshapeReshape4QNetwork/EncodingNetwork/conv2d_1/Relu:activations:0/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/flatten/Reshape?
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%QNetwork/EncodingNetwork/dense/MatMul?
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&QNetwork/EncodingNetwork/dense/BiasAdd?
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2%
#QNetwork/EncodingNetwork/dense/Relu?
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

: @*
dtype028
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?
'QNetwork/EncodingNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0>QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'QNetwork/EncodingNetwork/dense_1/MatMul?
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_1/MatMul:product:0?QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2*
(QNetwork/EncodingNetwork/dense_1/BiasAdd?
%QNetwork/EncodingNetwork/dense_1/ReluRelu1QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2'
%QNetwork/EncodingNetwork/dense_1/Relu?
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype028
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?
'QNetwork/EncodingNetwork/dense_2/MatMulMatMul3QNetwork/EncodingNetwork/dense_1/Relu:activations:0>QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'QNetwork/EncodingNetwork/dense_2/MatMul?
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_2/MatMul:product:0?QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_2/BiasAdd?
%QNetwork/EncodingNetwork/dense_2/ReluRelu1QNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2'
%QNetwork/EncodingNetwork/dense_2/Relu?
&QNetwork/dense_3/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&QNetwork/dense_3/MatMul/ReadVariableOp?
QNetwork/dense_3/MatMulMatMul3QNetwork/EncodingNetwork/dense_2/Relu:activations:0.QNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_3/MatMul?
'QNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'QNetwork/dense_3/BiasAdd/ReadVariableOp?
QNetwork/dense_3/BiasAddBiasAdd!QNetwork/dense_3/MatMul:product:0/QNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_3/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax!QNetwork/dense_3/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:07^QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp6^QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp6^QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5^QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp(^QNetwork/dense_3/BiasAdd/ReadVariableOp'^QNetwork/dense_3/MatMul/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesz
x:?????????:?????????:?????????:?????????::::::::::::2p
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp2n
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp2n
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2l
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2R
'QNetwork/dense_3/BiasAdd/ReadVariableOp'QNetwork/dense_3/BiasAdd/ReadVariableOp2P
&QNetwork/dense_3/MatMul/ReadVariableOp&QNetwork/dense_3/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:ZV
/
_output_shapes
:?????????
#
_user_specified_name	time_step
?
.
,__inference_function_with_signature_14824970?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_584732
PartitionedCall*
_input_shapes 
?
?
&__inference_signature_wrapper_14824940
discount
observation

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *5
f0R.
,__inference_function_with_signature_148249062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesz
x:?????????:?????????:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:^Z
/
_output_shapes
:?????????
'
_user_specified_name0/observation:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:PL
#
_output_shapes
:?????????
%
_user_specified_name0/step_type
?)
?
!__inference__traced_save_14825256
file_prefix'
#savev2_variable_read_readvariableop	E
Asavev2_qnetwork_encodingnetwork_conv2d_kernel_read_readvariableopC
?savev2_qnetwork_encodingnetwork_conv2d_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_conv2d_1_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_conv2d_1_bias_read_readvariableopD
@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableopB
>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableopF
Bsavev2_qnetwork_encodingnetwork_dense_1_kernel_read_readvariableopD
@savev2_qnetwork_encodingnetwork_dense_1_bias_read_readvariableopF
Bsavev2_qnetwork_encodingnetwork_dense_2_kernel_read_readvariableopD
@savev2_qnetwork_encodingnetwork_dense_2_bias_read_readvariableop6
2savev2_qnetwork_dense_3_kernel_read_readvariableop4
0savev2_qnetwork_dense_3_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopAsavev2_qnetwork_encodingnetwork_conv2d_kernel_read_readvariableop?savev2_qnetwork_encodingnetwork_conv2d_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_conv2d_1_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_conv2d_1_bias_read_readvariableop@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableop>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableopBsavev2_qnetwork_encodingnetwork_dense_1_kernel_read_readvariableop@savev2_qnetwork_encodingnetwork_dense_1_bias_read_readvariableopBsavev2_qnetwork_encodingnetwork_dense_2_kernel_read_readvariableop@savev2_qnetwork_encodingnetwork_dense_2_bias_read_readvariableop2savev2_qnetwork_dense_3_kernel_read_readvariableop0savev2_qnetwork_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes|
z: : : : :  : :	? : : @:@:	@?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :%!

_output_shapes
:	? : 

_output_shapes
: :$ 

_output_shapes

: @: 	

_output_shapes
:@:%
!

_output_shapes
:	@?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?
8
&__inference_get_initial_state_14824946

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
8
&__inference_signature_wrapper_14824952

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *5
f0R.
,__inference_function_with_signature_148249472
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?|
?
*__inference_polymorphic_action_fn_14825051
	step_type

reward
discount
observationB
>qnetwork_encodingnetwork_conv2d_conv2d_readvariableop_resourceC
?qnetwork_encodingnetwork_conv2d_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_conv2d_1_conv2d_readvariableop_resourceE
Aqnetwork_encodingnetwork_conv2d_1_biasadd_readvariableop_resourceA
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceB
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceC
?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resourceC
?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource3
/qnetwork_dense_3_matmul_readvariableop_resource4
0qnetwork_dense_3_biasadd_readvariableop_resource
identity??6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp?5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp?5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?'QNetwork/dense_3/BiasAdd/ReadVariableOp?&QNetwork/dense_3/MatMul/ReadVariableOp?
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp?
&QNetwork/EncodingNetwork/conv2d/Conv2DConv2Dobservation=QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2(
&QNetwork/EncodingNetwork/conv2d/Conv2D?
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp?
'QNetwork/EncodingNetwork/conv2d/BiasAddBiasAdd/QNetwork/EncodingNetwork/conv2d/Conv2D:output:0>QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2)
'QNetwork/EncodingNetwork/conv2d/BiasAdd?
$QNetwork/EncodingNetwork/conv2d/ReluRelu0QNetwork/EncodingNetwork/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2&
$QNetwork/EncodingNetwork/conv2d/Relu?
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype029
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp?
(QNetwork/EncodingNetwork/conv2d_1/Conv2DConv2D2QNetwork/EncodingNetwork/conv2d/Relu:activations:0?QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2*
(QNetwork/EncodingNetwork/conv2d_1/Conv2D?
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/conv2d_1/Conv2D:output:0@QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2+
)QNetwork/EncodingNetwork/conv2d_1/BiasAdd?
&QNetwork/EncodingNetwork/conv2d_1/ReluRelu2QNetwork/EncodingNetwork/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2(
&QNetwork/EncodingNetwork/conv2d_1/Relu?
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&QNetwork/EncodingNetwork/flatten/Const?
(QNetwork/EncodingNetwork/flatten/ReshapeReshape4QNetwork/EncodingNetwork/conv2d_1/Relu:activations:0/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/flatten/Reshape?
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%QNetwork/EncodingNetwork/dense/MatMul?
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&QNetwork/EncodingNetwork/dense/BiasAdd?
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2%
#QNetwork/EncodingNetwork/dense/Relu?
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

: @*
dtype028
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?
'QNetwork/EncodingNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0>QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'QNetwork/EncodingNetwork/dense_1/MatMul?
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_1/MatMul:product:0?QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2*
(QNetwork/EncodingNetwork/dense_1/BiasAdd?
%QNetwork/EncodingNetwork/dense_1/ReluRelu1QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2'
%QNetwork/EncodingNetwork/dense_1/Relu?
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype028
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?
'QNetwork/EncodingNetwork/dense_2/MatMulMatMul3QNetwork/EncodingNetwork/dense_1/Relu:activations:0>QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'QNetwork/EncodingNetwork/dense_2/MatMul?
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_2/MatMul:product:0?QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_2/BiasAdd?
%QNetwork/EncodingNetwork/dense_2/ReluRelu1QNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2'
%QNetwork/EncodingNetwork/dense_2/Relu?
&QNetwork/dense_3/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&QNetwork/dense_3/MatMul/ReadVariableOp?
QNetwork/dense_3/MatMulMatMul3QNetwork/EncodingNetwork/dense_2/Relu:activations:0.QNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_3/MatMul?
'QNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'QNetwork/dense_3/BiasAdd/ReadVariableOp?
QNetwork/dense_3/BiasAddBiasAdd!QNetwork/dense_3/MatMul:product:0/QNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_3/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax!QNetwork/dense_3/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:07^QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp6^QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp6^QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5^QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp(^QNetwork/dense_3/BiasAdd/ReadVariableOp'^QNetwork/dense_3/MatMul/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesz
x:?????????:?????????:?????????:?????????::::::::::::2p
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp2n
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp2n
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2l
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2R
'QNetwork/dense_3/BiasAdd/ReadVariableOp'QNetwork/dense_3/BiasAdd/ReadVariableOp2P
&QNetwork/dense_3/MatMul/ReadVariableOp&QNetwork/dense_3/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:\X
/
_output_shapes
:?????????
%
_user_specified_nameobservation
?
[
__inference_<lambda>_58470
readvariableop_resource
identity	??ReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpj
IdentityIdentityReadVariableOp:value:0^ReadVariableOp*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2 
ReadVariableOpReadVariableOp
?
>
,__inference_function_with_signature_14824947

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_get_initial_state_148249462
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
`
&__inference_signature_wrapper_14824967
unknown
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *5
f0R.
,__inference_function_with_signature_148249592
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
?b
?
0__inference_polymorphic_distribution_fn_14825186
	step_type

reward
discount
observationB
>qnetwork_encodingnetwork_conv2d_conv2d_readvariableop_resourceC
?qnetwork_encodingnetwork_conv2d_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_conv2d_1_conv2d_readvariableop_resourceE
Aqnetwork_encodingnetwork_conv2d_1_biasadd_readvariableop_resourceA
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceB
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceC
?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resourceC
?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource3
/qnetwork_dense_3_matmul_readvariableop_resource4
0qnetwork_dense_3_biasadd_readvariableop_resource
identity??6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp?5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp?5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?'QNetwork/dense_3/BiasAdd/ReadVariableOp?&QNetwork/dense_3/MatMul/ReadVariableOp?
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp?
&QNetwork/EncodingNetwork/conv2d/Conv2DConv2Dobservation=QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2(
&QNetwork/EncodingNetwork/conv2d/Conv2D?
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp?
'QNetwork/EncodingNetwork/conv2d/BiasAddBiasAdd/QNetwork/EncodingNetwork/conv2d/Conv2D:output:0>QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2)
'QNetwork/EncodingNetwork/conv2d/BiasAdd?
$QNetwork/EncodingNetwork/conv2d/ReluRelu0QNetwork/EncodingNetwork/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2&
$QNetwork/EncodingNetwork/conv2d/Relu?
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype029
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp?
(QNetwork/EncodingNetwork/conv2d_1/Conv2DConv2D2QNetwork/EncodingNetwork/conv2d/Relu:activations:0?QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2*
(QNetwork/EncodingNetwork/conv2d_1/Conv2D?
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/conv2d_1/Conv2D:output:0@QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2+
)QNetwork/EncodingNetwork/conv2d_1/BiasAdd?
&QNetwork/EncodingNetwork/conv2d_1/ReluRelu2QNetwork/EncodingNetwork/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2(
&QNetwork/EncodingNetwork/conv2d_1/Relu?
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&QNetwork/EncodingNetwork/flatten/Const?
(QNetwork/EncodingNetwork/flatten/ReshapeReshape4QNetwork/EncodingNetwork/conv2d_1/Relu:activations:0/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/flatten/Reshape?
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%QNetwork/EncodingNetwork/dense/MatMul?
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&QNetwork/EncodingNetwork/dense/BiasAdd?
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2%
#QNetwork/EncodingNetwork/dense/Relu?
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

: @*
dtype028
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?
'QNetwork/EncodingNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0>QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'QNetwork/EncodingNetwork/dense_1/MatMul?
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_1/MatMul:product:0?QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2*
(QNetwork/EncodingNetwork/dense_1/BiasAdd?
%QNetwork/EncodingNetwork/dense_1/ReluRelu1QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2'
%QNetwork/EncodingNetwork/dense_1/Relu?
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype028
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?
'QNetwork/EncodingNetwork/dense_2/MatMulMatMul3QNetwork/EncodingNetwork/dense_1/Relu:activations:0>QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'QNetwork/EncodingNetwork/dense_2/MatMul?
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_2/MatMul:product:0?QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_2/BiasAdd?
%QNetwork/EncodingNetwork/dense_2/ReluRelu1QNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2'
%QNetwork/EncodingNetwork/dense_2/Relu?
&QNetwork/dense_3/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&QNetwork/dense_3/MatMul/ReadVariableOp?
QNetwork/dense_3/MatMulMatMul3QNetwork/EncodingNetwork/dense_2/Relu:activations:0.QNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_3/MatMul?
'QNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'QNetwork/dense_3/BiasAdd/ReadVariableOp?
QNetwork/dense_3/BiasAddBiasAdd!QNetwork/dense_3/MatMul:product:0/QNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_3/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax!QNetwork/dense_3/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtoln
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/atoln
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/rtol?
IdentityIdentityCategorical_1/mode/Cast:y:07^QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp6^QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp6^QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5^QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp(^QNetwork/dense_3/BiasAdd/ReadVariableOp'^QNetwork/dense_3/MatMul/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identityn
Deterministic_2/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/atoln
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/rtol"
identityIdentity:output:0*?
_input_shapesz
x:?????????:?????????:?????????:?????????::::::::::::2p
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp2n
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp2n
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2l
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2R
'QNetwork/dense_3/BiasAdd/ReadVariableOp'QNetwork/dense_3/BiasAdd/ReadVariableOp2P
&QNetwork/dense_3/MatMul/ReadVariableOp&QNetwork/dense_3/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:\X
/
_output_shapes
:?????????
%
_user_specified_nameobservation
?<
?
$__inference__traced_restore_14825305
file_prefix
assignvariableop_variable=
9assignvariableop_1_qnetwork_encodingnetwork_conv2d_kernel;
7assignvariableop_2_qnetwork_encodingnetwork_conv2d_bias?
;assignvariableop_3_qnetwork_encodingnetwork_conv2d_1_kernel=
9assignvariableop_4_qnetwork_encodingnetwork_conv2d_1_bias<
8assignvariableop_5_qnetwork_encodingnetwork_dense_kernel:
6assignvariableop_6_qnetwork_encodingnetwork_dense_bias>
:assignvariableop_7_qnetwork_encodingnetwork_dense_1_kernel<
8assignvariableop_8_qnetwork_encodingnetwork_dense_1_bias>
:assignvariableop_9_qnetwork_encodingnetwork_dense_2_kernel=
9assignvariableop_10_qnetwork_encodingnetwork_dense_2_bias/
+assignvariableop_11_qnetwork_dense_3_kernel-
)assignvariableop_12_qnetwork_dense_3_bias
identity_14??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp9assignvariableop_1_qnetwork_encodingnetwork_conv2d_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp7assignvariableop_2_qnetwork_encodingnetwork_conv2d_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp;assignvariableop_3_qnetwork_encodingnetwork_conv2d_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp9assignvariableop_4_qnetwork_encodingnetwork_conv2d_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp8assignvariableop_5_qnetwork_encodingnetwork_dense_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp6assignvariableop_6_qnetwork_encodingnetwork_dense_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp:assignvariableop_7_qnetwork_encodingnetwork_dense_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp8assignvariableop_8_qnetwork_encodingnetwork_dense_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp:assignvariableop_9_qnetwork_encodingnetwork_dense_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp9assignvariableop_10_qnetwork_encodingnetwork_dense_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_qnetwork_dense_3_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_qnetwork_dense_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_13?
Identity_14IdentityIdentity_13:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_14"#
identity_14Identity_14:output:0*I
_input_shapes8
6: :::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
1

__inference_<lambda>_58473*
_input_shapes 
?
f
,__inference_function_with_signature_14824959
unknown
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_584702
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
?}
?
*__inference_polymorphic_action_fn_14825127
time_step_step_type
time_step_reward
time_step_discount
time_step_observationB
>qnetwork_encodingnetwork_conv2d_conv2d_readvariableop_resourceC
?qnetwork_encodingnetwork_conv2d_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_conv2d_1_conv2d_readvariableop_resourceE
Aqnetwork_encodingnetwork_conv2d_1_biasadd_readvariableop_resourceA
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceB
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceC
?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resourceC
?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource3
/qnetwork_dense_3_matmul_readvariableop_resource4
0qnetwork_dense_3_biasadd_readvariableop_resource
identity??6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp?5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp?5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?'QNetwork/dense_3/BiasAdd/ReadVariableOp?&QNetwork/dense_3/MatMul/ReadVariableOp?
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp?
&QNetwork/EncodingNetwork/conv2d/Conv2DConv2Dtime_step_observation=QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2(
&QNetwork/EncodingNetwork/conv2d/Conv2D?
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp?
'QNetwork/EncodingNetwork/conv2d/BiasAddBiasAdd/QNetwork/EncodingNetwork/conv2d/Conv2D:output:0>QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2)
'QNetwork/EncodingNetwork/conv2d/BiasAdd?
$QNetwork/EncodingNetwork/conv2d/ReluRelu0QNetwork/EncodingNetwork/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2&
$QNetwork/EncodingNetwork/conv2d/Relu?
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype029
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp?
(QNetwork/EncodingNetwork/conv2d_1/Conv2DConv2D2QNetwork/EncodingNetwork/conv2d/Relu:activations:0?QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2*
(QNetwork/EncodingNetwork/conv2d_1/Conv2D?
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/conv2d_1/Conv2D:output:0@QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2+
)QNetwork/EncodingNetwork/conv2d_1/BiasAdd?
&QNetwork/EncodingNetwork/conv2d_1/ReluRelu2QNetwork/EncodingNetwork/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2(
&QNetwork/EncodingNetwork/conv2d_1/Relu?
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&QNetwork/EncodingNetwork/flatten/Const?
(QNetwork/EncodingNetwork/flatten/ReshapeReshape4QNetwork/EncodingNetwork/conv2d_1/Relu:activations:0/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/flatten/Reshape?
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%QNetwork/EncodingNetwork/dense/MatMul?
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&QNetwork/EncodingNetwork/dense/BiasAdd?
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2%
#QNetwork/EncodingNetwork/dense/Relu?
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

: @*
dtype028
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?
'QNetwork/EncodingNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0>QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'QNetwork/EncodingNetwork/dense_1/MatMul?
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_1/MatMul:product:0?QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2*
(QNetwork/EncodingNetwork/dense_1/BiasAdd?
%QNetwork/EncodingNetwork/dense_1/ReluRelu1QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2'
%QNetwork/EncodingNetwork/dense_1/Relu?
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype028
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?
'QNetwork/EncodingNetwork/dense_2/MatMulMatMul3QNetwork/EncodingNetwork/dense_1/Relu:activations:0>QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'QNetwork/EncodingNetwork/dense_2/MatMul?
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_2/MatMul:product:0?QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_2/BiasAdd?
%QNetwork/EncodingNetwork/dense_2/ReluRelu1QNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2'
%QNetwork/EncodingNetwork/dense_2/Relu?
&QNetwork/dense_3/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&QNetwork/dense_3/MatMul/ReadVariableOp?
QNetwork/dense_3/MatMulMatMul3QNetwork/EncodingNetwork/dense_2/Relu:activations:0.QNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_3/MatMul?
'QNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'QNetwork/dense_3/BiasAdd/ReadVariableOp?
QNetwork/dense_3/BiasAddBiasAdd!QNetwork/dense_3/MatMul:product:0/QNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_3/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax!QNetwork/dense_3/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:07^QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp6^QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp6^QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5^QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp(^QNetwork/dense_3/BiasAdd/ReadVariableOp'^QNetwork/dense_3/MatMul/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesz
x:?????????:?????????:?????????:?????????::::::::::::2p
6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp6QNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp2n
5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp5QNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp7QNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp2n
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2l
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2R
'QNetwork/dense_3/BiasAdd/ReadVariableOp'QNetwork/dense_3/BiasAdd/ReadVariableOp2P
&QNetwork/dense_3/MatMul/ReadVariableOp&QNetwork/dense_3/MatMul/ReadVariableOp:X T
#
_output_shapes
:?????????
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:?????????
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:?????????
,
_user_specified_nametime_step/discount:fb
/
_output_shapes
:?????????
/
_user_specified_nametime_step/observation
?
8
&__inference_get_initial_state_14825189

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
(
&__inference_signature_wrapper_14824974?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *5
f0R.
,__inference_function_with_signature_148249702
PartitionedCall*
_input_shapes 
?
?
,__inference_function_with_signature_14824906
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *3
f.R,
*__inference_polymorphic_action_fn_148248792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesz
x:?????????:?????????:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:?????????
%
_user_specified_name0/step_type:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:OK
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:^Z
/
_output_shapes
:?????????
'
_user_specified_name0/observation"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
action?
4

0/discount&
action_0/discount:0?????????
F
0/observation5
action_0/observation:0?????????
0
0/reward$
action_0/reward:0?????????
6
0/step_type'
action_0/step_type:0?????????6
action,
StatefulPartitionedCall:0?????????tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:??
?

train_step
metadata
model_variables
_all_assets

signatures

oaction
pdistribution
qget_initial_state
rget_metadata
sget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
w
0
1
2
	3

4
5
6
7
8
9
10
11"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
`

taction
uget_initial_state
vget_train_step
wget_metadata"
signature_map
@:> 2&QNetwork/EncodingNetwork/conv2d/kernel
2:0 2$QNetwork/EncodingNetwork/conv2d/bias
B:@  2(QNetwork/EncodingNetwork/conv2d_1/kernel
4:2 2&QNetwork/EncodingNetwork/conv2d_1/bias
8:6	? 2%QNetwork/EncodingNetwork/dense/kernel
1:/ 2#QNetwork/EncodingNetwork/dense/bias
9:7 @2'QNetwork/EncodingNetwork/dense_1/kernel
3:1@2%QNetwork/EncodingNetwork/dense_1/bias
::8	@?2'QNetwork/EncodingNetwork/dense_2/kernel
4:2?2%QNetwork/EncodingNetwork/dense_2/bias
*:(	?2QNetwork/dense_3/kernel
#:!2QNetwork/dense_3/bias
1
ref
1"
trackable_tuple_wrapper
.

_q_network"
_generic_user_object
?
_encoder
_q_value_layer
	variables
regularization_losses
trainable_variables
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "QNetwork", "name": "QNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
_postprocessing_layers
	variables
regularization_losses
trainable_variables
	keras_api
z__call__
*{&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
|__call__
*}&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
v
0
1
2
	3

4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
	3

4
5
6
7
8
9
10
11"
trackable_list_wrapper
?
	variables
regularization_losses
$non_trainable_variables

%layers
&layer_metrics
'metrics
(layer_regularization_losses
trainable_variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
J
)0
*1
+2
,3
-4
.5"
trackable_list_wrapper
f
0
1
2
	3

4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
	3

4
5
6
7
8
9"
trackable_list_wrapper
?
	variables
regularization_losses
/non_trainable_variables

0layers
1layer_metrics
2metrics
3layer_regularization_losses
trainable_variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
 	variables
!regularization_losses
4non_trainable_variables

5layers
6layer_metrics
7metrics
8layer_regularization_losses
"trainable_variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?


kernel
bias
9	variables
:regularization_losses
;trainable_variables
<	keras_api
~__call__
*&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 17, 17, 4]}}
?


kernel
	bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 10, 10, 32]}}
?
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?


kernel
bias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1568}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1568]}}
?

kernel
bias
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
?

kernel
bias
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 64]}}
 "
trackable_list_wrapper
J
)0
*1
+2
,3
-4
.5"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
9	variables
:regularization_losses
Qnon_trainable_variables

Rlayers
Slayer_metrics
Tmetrics
Ulayer_regularization_losses
;trainable_variables
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
?
=	variables
>regularization_losses
Vnon_trainable_variables

Wlayers
Xlayer_metrics
Ymetrics
Zlayer_regularization_losses
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
A	variables
Bregularization_losses
[non_trainable_variables

\layers
]layer_metrics
^metrics
_layer_regularization_losses
Ctrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?
E	variables
Fregularization_losses
`non_trainable_variables

alayers
blayer_metrics
cmetrics
dlayer_regularization_losses
Gtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
I	variables
Jregularization_losses
enon_trainable_variables

flayers
glayer_metrics
hmetrics
ilayer_regularization_losses
Ktrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
M	variables
Nregularization_losses
jnon_trainable_variables

klayers
llayer_metrics
mmetrics
nlayer_regularization_losses
Otrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
*__inference_polymorphic_action_fn_14825051
*__inference_polymorphic_action_fn_14825127?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_polymorphic_distribution_fn_14825186?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_get_initial_state_14825189?
???
FullArgSpec!
args?
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_58473"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_58470"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_14824940
0/discount0/observation0/reward0/step_type"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_14824952
batch_size"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_14824967"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_14824974"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 9
__inference_<lambda>_58470?

? 
? "? 	2
__inference_<lambda>_58473?

? 
? "? S
&__inference_get_initial_state_14825189)"?
?
?

batch_size 
? "? ?
*__inference_polymorphic_action_fn_14825051?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????<
observation-?*
observation?????????
? 
? "R?O

PolicyStep&
action?
action?????????
state? 
info? ?
*__inference_polymorphic_action_fn_14825127?	
???
???
???
TimeStep6
	step_type)?&
time_step/step_type?????????0
reward&?#
time_step/reward?????????4
discount(?%
time_step/discount?????????F
observation7?4
time_step/observation?????????
? 
? "R?O

PolicyStep&
action?
action?????????
state? 
info? ?
0__inference_polymorphic_distribution_fn_14825186?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????<
observation-?*
observation?????????
? 
? "???

PolicyStep?
action?????Ã}?z
`
C?@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
*?'
%
loc?
Identity?????????
? _TFPTypeSpec
state? 
info? ?
&__inference_signature_wrapper_14824940?	
???
? 
???
.

0/discount ?

0/discount?????????
@
0/observation/?,
0/observation?????????
*
0/reward?
0/reward?????????
0
0/step_type!?
0/step_type?????????"+?(
&
action?
action?????????a
&__inference_signature_wrapper_1482495270?-
? 
&?#
!

batch_size?

batch_size "? Z
&__inference_signature_wrapper_148249670?

? 
? "?

int64?
int64 	>
&__inference_signature_wrapper_14824974?

? 
? "? 
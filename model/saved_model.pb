??
?&?&
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
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
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
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
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
?
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
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
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
?
embedding_12/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*(
shared_nameembedding_12/embeddings
?
+embedding_12/embeddings/Read/ReadVariableOpReadVariableOpembedding_12/embeddings*
_output_shapes
:	?N*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:*
dtype0
?
string_lookup_12_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_286737*
value_dtype0	
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
?
Adam/embedding_12/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*/
shared_name Adam/embedding_12/embeddings/m
?
2Adam/embedding_12/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_12/embeddings/m*
_output_shapes
:	?N*
dtype0
?
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_12/kernel/m
?
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_12/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*/
shared_name Adam/embedding_12/embeddings/v
?
2Adam/embedding_12/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_12/embeddings/v*
_output_shapes
:	?N*
dtype0
?
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_12/kernel/v
?
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *$
fR
__inference_<lambda>_311080

NoOpNoOp^PartitionedCall
?
Kstring_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_12_index_table*
Tkeys0*
Tvalues0	*/
_class%
#!loc:@string_lookup_12_index_table*
_output_shapes

::
?.
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*?-
value?-B?- B?-
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
=

state_variables
_index_lookup_layer
	keras_api
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
 

0
1
2
 

1
2
3
?
layer_regularization_losses
trainable_variables
layer_metrics
 non_trainable_variables

!layers
regularization_losses
"metrics
	variables
 
 
0
#state_variables

$_table
%	keras_api
 
b

embeddings
&trainable_variables
'regularization_losses
(	variables
)	keras_api
R
*trainable_variables
+regularization_losses
,	variables
-	keras_api
R
.trainable_variables
/regularization_losses
0	variables
1	keras_api
R
2trainable_variables
3regularization_losses
4	variables
5	keras_api
h

kernel
bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
v
:iter

;beta_1

<beta_2
	=decay
>learning_ratemxmymzv{v|v}

0
1
2
 

0
1
2
?
?layer_regularization_losses
trainable_variables
@layer_metrics
Anon_trainable_variables

Blayers
regularization_losses
Cmetrics
	variables
 
 
 
?
Dlayer_regularization_losses
trainable_variables
Elayer_metrics
Fnon_trainable_variables

Glayers
regularization_losses
Hmetrics
	variables
][
VARIABLE_VALUEembedding_12/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_12/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_12/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2

I0
J1
 
LJ
tableAlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table
 

0
 

0
?
Klayer_regularization_losses
&trainable_variables
Llayer_metrics
Mnon_trainable_variables

Nlayers
'regularization_losses
Ometrics
(	variables
 
 
 
?
Player_regularization_losses
*trainable_variables
Qlayer_metrics
Rnon_trainable_variables

Slayers
+regularization_losses
Tmetrics
,	variables
 
 
 
?
Ulayer_regularization_losses
.trainable_variables
Vlayer_metrics
Wnon_trainable_variables

Xlayers
/regularization_losses
Ymetrics
0	variables
 
 
 
?
Zlayer_regularization_losses
2trainable_variables
[layer_metrics
\non_trainable_variables

]layers
3regularization_losses
^metrics
4	variables

0
1
 

0
1
?
_layer_regularization_losses
6trainable_variables
`layer_metrics
anon_trainable_variables

blayers
7regularization_losses
cmetrics
8	variables
][
VARIABLE_VALUE	Adam/iter>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEAdam/beta_1@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEAdam/beta_2@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE
Adam/decay?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/learning_rateGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
#
0
1
2
3
4

d0
e1
 
 
 
 
 
4
	ftotal
	gcount
h	variables
i	keras_api
D
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api
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
4
	ototal
	pcount
q	variables
r	keras_api
D
	stotal
	tcount
u
_fn_kwargs
v	variables
w	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

h	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

m	variables
fd
VARIABLE_VALUEtotal_2Ilayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEcount_2Ilayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

o0
p1

q	variables
fd
VARIABLE_VALUEtotal_3Ilayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEcount_3Ilayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

s0
t1

v	variables
??
VARIABLE_VALUEAdam/embedding_12/embeddings/matrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_12/kernel/matrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_12/bias/matrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_12/embeddings/vatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_12/kernel/vatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_12/bias/vatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
+serving_default_text_vectorization_12_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall+serving_default_text_vectorization_12_inputstring_lookup_12_index_tableConstembedding_12/embeddingsdense_12/kerneldense_12/bias*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_310482
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+embedding_12/embeddings/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOpKstring_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2Mstring_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2:1Adam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp2Adam/embedding_12/embeddings/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp2Adam/embedding_12/embeddings/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOpConst_1*%
Tin
2		*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_311176
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_12/embeddingsdense_12/kerneldense_12/biasstring_lookup_12_index_table	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3Adam/embedding_12/embeddings/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/embedding_12/embeddings/vAdam/dense_12/kernel/vAdam/dense_12/bias/v*#
Tin
2*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_311255Ƨ
?-
?
I__inference_sequential_23_layer_call_and_return_conditional_losses_310792

inputs	(
$embedding_12_embedding_lookup_310762+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?embedding_12/embedding_lookup^
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????2
Castz
embedding_12/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:??????????2
embedding_12/Cast?
embedding_12/embedding_lookupResourceGather$embedding_12_embedding_lookup_310762embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_12/embedding_lookup/310762*,
_output_shapes
:??????????*
dtype02
embedding_12/embedding_lookup?
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_12/embedding_lookup/310762*,
_output_shapes
:??????????2(
&embedding_12/embedding_lookup/Identity?
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_12/embedding_lookup/Identity_1y
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_24/dropout/Const?
dropout_24/dropout/MulMul1embedding_12/embedding_lookup/Identity_1:output:0!dropout_24/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_24/dropout/Mul?
dropout_24/dropout/ShapeShape1embedding_12/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dropout_24/dropout/Shape?
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_24/dropout/random_uniform/RandomUniform?
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_24/dropout/GreaterEqual/y?
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_24/dropout/GreaterEqual?
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_24/dropout/Cast?
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_24/dropout/Mul_1?
2global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2global_average_pooling1d_12/Mean/reduction_indices?
 global_average_pooling1d_12/MeanMeandropout_24/dropout/Mul_1:z:0;global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2"
 global_average_pooling1d_12/Meany
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_25/dropout/Const?
dropout_25/dropout/MulMul)global_average_pooling1d_12/Mean:output:0!dropout_25/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_25/dropout/Mul?
dropout_25/dropout/ShapeShape)global_average_pooling1d_12/Mean:output:0*
T0*
_output_shapes
:2
dropout_25/dropout/Shape?
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype021
/dropout_25/dropout/random_uniform/RandomUniform?
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_25/dropout/GreaterEqual/y?
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2!
dropout_25/dropout/GreaterEqual?
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_25/dropout/Cast?
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_25/dropout/Mul_1?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_25/dropout/Mul_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd?
IdentityIdentitydense_12/BiasAdd:output:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp^embedding_12/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????:::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2>
embedding_12/embedding_lookupembedding_12/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_23_layer_call_fn_310824

inputs	
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_3101052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_12_layer_call_fn_311033

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3099062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_24_layer_call_fn_310734

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_3103492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
.__inference_sequential_23_layer_call_fn_310835

inputs	
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_3101262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_23_layer_call_fn_309965
embedding_12_input
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_12_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_3099562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:??????????????????
,
_user_specified_nameembedding_12_input
ގ
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_310170
text_vectorization_12_input`
\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handlea
]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value	
sequential_23_310149
sequential_23_310151
sequential_23_310153
identity??%sequential_23/StatefulPartitionedCall?Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
!text_vectorization_12/StringLowerStringLowertext_vectorization_12_input*'
_output_shapes
:?????????2#
!text_vectorization_12/StringLower?
(text_vectorization_12/StaticRegexReplaceStaticRegexReplace*text_vectorization_12/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2*
(text_vectorization_12/StaticRegexReplace?
*text_vectorization_12/StaticRegexReplace_1StaticRegexReplace1text_vectorization_12/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2,
*text_vectorization_12/StaticRegexReplace_1?
text_vectorization_12/SqueezeSqueeze3text_vectorization_12/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_12/Squeeze?
'text_vectorization_12/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2)
'text_vectorization_12/StringSplit/Const?
/text_vectorization_12/StringSplit/StringSplitV2StringSplitV2&text_vectorization_12/Squeeze:output:00text_vectorization_12/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:21
/text_vectorization_12/StringSplit/StringSplitV2?
5text_vectorization_12/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5text_vectorization_12/StringSplit/strided_slice/stack?
7text_vectorization_12/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7text_vectorization_12/StringSplit/strided_slice/stack_1?
7text_vectorization_12/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7text_vectorization_12/StringSplit/strided_slice/stack_2?
/text_vectorization_12/StringSplit/strided_sliceStridedSlice9text_vectorization_12/StringSplit/StringSplitV2:indices:0>text_vectorization_12/StringSplit/strided_slice/stack:output:0@text_vectorization_12/StringSplit/strided_slice/stack_1:output:0@text_vectorization_12/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask21
/text_vectorization_12/StringSplit/strided_slice?
7text_vectorization_12/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7text_vectorization_12/StringSplit/strided_slice_1/stack?
9text_vectorization_12/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_1?
9text_vectorization_12/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_2?
1text_vectorization_12/StringSplit/strided_slice_1StridedSlice7text_vectorization_12/StringSplit/StringSplitV2:shape:0@text_vectorization_12/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask23
1text_vectorization_12/StringSplit/strided_slice_1?
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_12/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Z
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_12/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2h
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2g
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2e
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2LookupTableFindV2\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handle8text_vectorization_12/StringSplit/StringSplitV2:values:0]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2Q
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
8text_vectorization_12/string_lookup_12/assert_equal/NoOpNoOp*
_output_shapes
 2:
8text_vectorization_12/string_lookup_12/assert_equal/NoOp?
/text_vectorization_12/string_lookup_12/IdentityIdentityXtext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_12/string_lookup_12/Identity?
1text_vectorization_12/string_lookup_12/Identity_1Identityctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????23
1text_vectorization_12/string_lookup_12/Identity_1?
2text_vectorization_12/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 24
2text_vectorization_12/RaggedToTensor/default_value?
*text_vectorization_12/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2,
*text_vectorization_12/RaggedToTensor/Const?
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_12/RaggedToTensor/Const:output:08text_vectorization_12/string_lookup_12/Identity:output:0;text_vectorization_12/RaggedToTensor/default_value:output:0:text_vectorization_12/string_lookup_12/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2;
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_12/ShapeShapeBtext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_12/Shape?
)text_vectorization_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)text_vectorization_12/strided_slice/stack?
+text_vectorization_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_1?
+text_vectorization_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_2?
#text_vectorization_12/strided_sliceStridedSlice$text_vectorization_12/Shape:output:02text_vectorization_12/strided_slice/stack:output:04text_vectorization_12/strided_slice/stack_1:output:04text_vectorization_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#text_vectorization_12/strided_slice}
text_vectorization_12/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/sub/x?
text_vectorization_12/subSub$text_vectorization_12/sub/x:output:0,text_vectorization_12/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/sub
text_vectorization_12/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/Less/y?
text_vectorization_12/LessLess,text_vectorization_12/strided_slice:output:0%text_vectorization_12/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/Less?
text_vectorization_12/condStatelessIftext_vectorization_12/Less:z:0text_vectorization_12/sub:z:0Btext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'text_vectorization_12_cond_false_310051*/
output_shapes
:??????????????????*9
then_branch*R(
&text_vectorization_12_cond_true_3100502
text_vectorization_12/cond?
#text_vectorization_12/cond/IdentityIdentity#text_vectorization_12/cond:output:0*
T0	*(
_output_shapes
:??????????2%
#text_vectorization_12/cond/Identity?
%sequential_23/StatefulPartitionedCallStatefulPartitionedCall,text_vectorization_12/cond/Identity:output:0sequential_23_310149sequential_23_310151sequential_23_310153*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_3101052'
%sequential_23/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall.sequential_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_11_layer_call_and_return_conditional_losses_3101612
activation_11/PartitionedCall?
IdentityIdentity&activation_11/PartitionedCall:output:0&^sequential_23/StatefulPartitionedCallP^text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::2N
%sequential_23/StatefulPartitionedCall%sequential_23/StatefulPartitionedCall2?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:d `
'
_output_shapes
:?????????
5
_user_specified_nametext_vectorization_12_input:

_output_shapes
: 
?
?
.__inference_sequential_23_layer_call_fn_309991
embedding_12_input
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_12_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_3099822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:??????????????????
,
_user_specified_nameembedding_12_input
?
-
__inference__destroyer_311048
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
e
I__inference_activation_11_layer_call_and_return_conditional_losses_310161

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'text_vectorization_12_cond_false_310051*
&text_vectorization_12_cond_placeholderf
btext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
.text_vectorization_12/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.text_vectorization_12/cond/strided_slice/stack?
0text_vectorization_12/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   22
0text_vectorization_12/cond/strided_slice/stack_1?
0text_vectorization_12/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0text_vectorization_12/cond/strided_slice/stack_2?
(text_vectorization_12/cond/strided_sliceStridedSlicebtext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor7text_vectorization_12/cond/strided_slice/stack:output:09text_vectorization_12/cond/strided_slice/stack_1:output:09text_vectorization_12/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2*
(text_vectorization_12/cond/strided_slice?
#text_vectorization_12/cond/IdentityIdentity1text_vectorization_12/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
s
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_309859

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_310452

inputs`
\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handlea
]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value	
sequential_23_310443
sequential_23_310445
sequential_23_310447
identity??%sequential_23/StatefulPartitionedCall?Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
!text_vectorization_12/StringLowerStringLowerinputs*'
_output_shapes
:?????????2#
!text_vectorization_12/StringLower?
(text_vectorization_12/StaticRegexReplaceStaticRegexReplace*text_vectorization_12/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2*
(text_vectorization_12/StaticRegexReplace?
*text_vectorization_12/StaticRegexReplace_1StaticRegexReplace1text_vectorization_12/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2,
*text_vectorization_12/StaticRegexReplace_1?
text_vectorization_12/SqueezeSqueeze3text_vectorization_12/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_12/Squeeze?
'text_vectorization_12/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2)
'text_vectorization_12/StringSplit/Const?
/text_vectorization_12/StringSplit/StringSplitV2StringSplitV2&text_vectorization_12/Squeeze:output:00text_vectorization_12/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:21
/text_vectorization_12/StringSplit/StringSplitV2?
5text_vectorization_12/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5text_vectorization_12/StringSplit/strided_slice/stack?
7text_vectorization_12/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7text_vectorization_12/StringSplit/strided_slice/stack_1?
7text_vectorization_12/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7text_vectorization_12/StringSplit/strided_slice/stack_2?
/text_vectorization_12/StringSplit/strided_sliceStridedSlice9text_vectorization_12/StringSplit/StringSplitV2:indices:0>text_vectorization_12/StringSplit/strided_slice/stack:output:0@text_vectorization_12/StringSplit/strided_slice/stack_1:output:0@text_vectorization_12/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask21
/text_vectorization_12/StringSplit/strided_slice?
7text_vectorization_12/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7text_vectorization_12/StringSplit/strided_slice_1/stack?
9text_vectorization_12/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_1?
9text_vectorization_12/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_2?
1text_vectorization_12/StringSplit/strided_slice_1StridedSlice7text_vectorization_12/StringSplit/StringSplitV2:shape:0@text_vectorization_12/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask23
1text_vectorization_12/StringSplit/strided_slice_1?
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_12/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Z
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_12/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2h
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2g
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2e
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2LookupTableFindV2\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handle8text_vectorization_12/StringSplit/StringSplitV2:values:0]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2Q
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
8text_vectorization_12/string_lookup_12/assert_equal/NoOpNoOp*
_output_shapes
 2:
8text_vectorization_12/string_lookup_12/assert_equal/NoOp?
/text_vectorization_12/string_lookup_12/IdentityIdentityXtext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_12/string_lookup_12/Identity?
1text_vectorization_12/string_lookup_12/Identity_1Identityctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????23
1text_vectorization_12/string_lookup_12/Identity_1?
2text_vectorization_12/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 24
2text_vectorization_12/RaggedToTensor/default_value?
*text_vectorization_12/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2,
*text_vectorization_12/RaggedToTensor/Const?
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_12/RaggedToTensor/Const:output:08text_vectorization_12/string_lookup_12/Identity:output:0;text_vectorization_12/RaggedToTensor/default_value:output:0:text_vectorization_12/string_lookup_12/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2;
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_12/ShapeShapeBtext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_12/Shape?
)text_vectorization_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)text_vectorization_12/strided_slice/stack?
+text_vectorization_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_1?
+text_vectorization_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_2?
#text_vectorization_12/strided_sliceStridedSlice$text_vectorization_12/Shape:output:02text_vectorization_12/strided_slice/stack:output:04text_vectorization_12/strided_slice/stack_1:output:04text_vectorization_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#text_vectorization_12/strided_slice}
text_vectorization_12/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/sub/x?
text_vectorization_12/subSub$text_vectorization_12/sub/x:output:0,text_vectorization_12/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/sub
text_vectorization_12/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/Less/y?
text_vectorization_12/LessLess,text_vectorization_12/strided_slice:output:0%text_vectorization_12/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/Less?
text_vectorization_12/condStatelessIftext_vectorization_12/Less:z:0text_vectorization_12/sub:z:0Btext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'text_vectorization_12_cond_false_310423*/
output_shapes
:??????????????????*9
then_branch*R(
&text_vectorization_12_cond_true_3104222
text_vectorization_12/cond?
#text_vectorization_12/cond/IdentityIdentity#text_vectorization_12/cond:output:0*
T0	*(
_output_shapes
:??????????2%
#text_vectorization_12/cond/Identity?
%sequential_23/StatefulPartitionedCallStatefulPartitionedCall,text_vectorization_12/cond/Identity:output:0sequential_23_310443sequential_23_310445sequential_23_310447*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_3101262'
%sequential_23/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall.sequential_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_11_layer_call_and_return_conditional_losses_3101612
activation_11/PartitionedCall?
IdentityIdentity&activation_11/PartitionedCall:output:0&^sequential_23/StatefulPartitionedCallP^text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::2N
%sequential_23/StatefulPartitionedCall%sequential_23/StatefulPartitionedCall2?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
4sequential_24_text_vectorization_12_cond_true_309739c
_sequential_24_text_vectorization_12_cond_pad_paddings_1_sequential_24_text_vectorization_12_subx
tsequential_24_text_vectorization_12_cond_pad_sequential_24_text_vectorization_12_raggedtotensor_raggedtensortotensor	5
1sequential_24_text_vectorization_12_cond_identity	?
9sequential_24/text_vectorization_12/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2;
9sequential_24/text_vectorization_12/cond/Pad/paddings/1/0?
7sequential_24/text_vectorization_12/cond/Pad/paddings/1PackBsequential_24/text_vectorization_12/cond/Pad/paddings/1/0:output:0_sequential_24_text_vectorization_12_cond_pad_paddings_1_sequential_24_text_vectorization_12_sub*
N*
T0*
_output_shapes
:29
7sequential_24/text_vectorization_12/cond/Pad/paddings/1?
9sequential_24/text_vectorization_12/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9sequential_24/text_vectorization_12/cond/Pad/paddings/0_1?
5sequential_24/text_vectorization_12/cond/Pad/paddingsPackBsequential_24/text_vectorization_12/cond/Pad/paddings/0_1:output:0@sequential_24/text_vectorization_12/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:27
5sequential_24/text_vectorization_12/cond/Pad/paddings?
,sequential_24/text_vectorization_12/cond/PadPadtsequential_24_text_vectorization_12_cond_pad_sequential_24_text_vectorization_12_raggedtotensor_raggedtensortotensor>sequential_24/text_vectorization_12/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2.
,sequential_24/text_vectorization_12/cond/Pad?
1sequential_24/text_vectorization_12/cond/IdentityIdentity5sequential_24/text_vectorization_12/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????23
1sequential_24/text_vectorization_12/cond/Identity"o
1sequential_24_text_vectorization_12_cond_identity:sequential_24/text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
??
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_310621

inputs`
\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handlea
]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value	6
2sequential_23_embedding_12_embedding_lookup_3105909
5sequential_23_dense_12_matmul_readvariableop_resource:
6sequential_23_dense_12_biasadd_readvariableop_resource
identity??-sequential_23/dense_12/BiasAdd/ReadVariableOp?,sequential_23/dense_12/MatMul/ReadVariableOp?+sequential_23/embedding_12/embedding_lookup?Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
!text_vectorization_12/StringLowerStringLowerinputs*'
_output_shapes
:?????????2#
!text_vectorization_12/StringLower?
(text_vectorization_12/StaticRegexReplaceStaticRegexReplace*text_vectorization_12/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2*
(text_vectorization_12/StaticRegexReplace?
*text_vectorization_12/StaticRegexReplace_1StaticRegexReplace1text_vectorization_12/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2,
*text_vectorization_12/StaticRegexReplace_1?
text_vectorization_12/SqueezeSqueeze3text_vectorization_12/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_12/Squeeze?
'text_vectorization_12/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2)
'text_vectorization_12/StringSplit/Const?
/text_vectorization_12/StringSplit/StringSplitV2StringSplitV2&text_vectorization_12/Squeeze:output:00text_vectorization_12/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:21
/text_vectorization_12/StringSplit/StringSplitV2?
5text_vectorization_12/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5text_vectorization_12/StringSplit/strided_slice/stack?
7text_vectorization_12/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7text_vectorization_12/StringSplit/strided_slice/stack_1?
7text_vectorization_12/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7text_vectorization_12/StringSplit/strided_slice/stack_2?
/text_vectorization_12/StringSplit/strided_sliceStridedSlice9text_vectorization_12/StringSplit/StringSplitV2:indices:0>text_vectorization_12/StringSplit/strided_slice/stack:output:0@text_vectorization_12/StringSplit/strided_slice/stack_1:output:0@text_vectorization_12/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask21
/text_vectorization_12/StringSplit/strided_slice?
7text_vectorization_12/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7text_vectorization_12/StringSplit/strided_slice_1/stack?
9text_vectorization_12/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_1?
9text_vectorization_12/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_2?
1text_vectorization_12/StringSplit/strided_slice_1StridedSlice7text_vectorization_12/StringSplit/StringSplitV2:shape:0@text_vectorization_12/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask23
1text_vectorization_12/StringSplit/strided_slice_1?
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_12/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Z
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_12/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2h
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2g
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2e
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2LookupTableFindV2\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handle8text_vectorization_12/StringSplit/StringSplitV2:values:0]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2Q
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
8text_vectorization_12/string_lookup_12/assert_equal/NoOpNoOp*
_output_shapes
 2:
8text_vectorization_12/string_lookup_12/assert_equal/NoOp?
/text_vectorization_12/string_lookup_12/IdentityIdentityXtext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_12/string_lookup_12/Identity?
1text_vectorization_12/string_lookup_12/Identity_1Identityctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????23
1text_vectorization_12/string_lookup_12/Identity_1?
2text_vectorization_12/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 24
2text_vectorization_12/RaggedToTensor/default_value?
*text_vectorization_12/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2,
*text_vectorization_12/RaggedToTensor/Const?
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_12/RaggedToTensor/Const:output:08text_vectorization_12/string_lookup_12/Identity:output:0;text_vectorization_12/RaggedToTensor/default_value:output:0:text_vectorization_12/string_lookup_12/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2;
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_12/ShapeShapeBtext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_12/Shape?
)text_vectorization_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)text_vectorization_12/strided_slice/stack?
+text_vectorization_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_1?
+text_vectorization_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_2?
#text_vectorization_12/strided_sliceStridedSlice$text_vectorization_12/Shape:output:02text_vectorization_12/strided_slice/stack:output:04text_vectorization_12/strided_slice/stack_1:output:04text_vectorization_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#text_vectorization_12/strided_slice}
text_vectorization_12/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/sub/x?
text_vectorization_12/subSub$text_vectorization_12/sub/x:output:0,text_vectorization_12/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/sub
text_vectorization_12/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/Less/y?
text_vectorization_12/LessLess,text_vectorization_12/strided_slice:output:0%text_vectorization_12/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/Less?
text_vectorization_12/condStatelessIftext_vectorization_12/Less:z:0text_vectorization_12/sub:z:0Btext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'text_vectorization_12_cond_false_310568*/
output_shapes
:??????????????????*9
then_branch*R(
&text_vectorization_12_cond_true_3105672
text_vectorization_12/cond?
#text_vectorization_12/cond/IdentityIdentity#text_vectorization_12/cond:output:0*
T0	*(
_output_shapes
:??????????2%
#text_vectorization_12/cond/Identity?
sequential_23/CastCast,text_vectorization_12/cond/Identity:output:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????2
sequential_23/Cast?
sequential_23/embedding_12/CastCastsequential_23/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:??????????2!
sequential_23/embedding_12/Cast?
+sequential_23/embedding_12/embedding_lookupResourceGather2sequential_23_embedding_12_embedding_lookup_310590#sequential_23/embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*E
_class;
97loc:@sequential_23/embedding_12/embedding_lookup/310590*,
_output_shapes
:??????????*
dtype02-
+sequential_23/embedding_12/embedding_lookup?
4sequential_23/embedding_12/embedding_lookup/IdentityIdentity4sequential_23/embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@sequential_23/embedding_12/embedding_lookup/310590*,
_output_shapes
:??????????26
4sequential_23/embedding_12/embedding_lookup/Identity?
6sequential_23/embedding_12/embedding_lookup/Identity_1Identity=sequential_23/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????28
6sequential_23/embedding_12/embedding_lookup/Identity_1?
&sequential_23/dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&sequential_23/dropout_24/dropout/Const?
$sequential_23/dropout_24/dropout/MulMul?sequential_23/embedding_12/embedding_lookup/Identity_1:output:0/sequential_23/dropout_24/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2&
$sequential_23/dropout_24/dropout/Mul?
&sequential_23/dropout_24/dropout/ShapeShape?sequential_23/embedding_12/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2(
&sequential_23/dropout_24/dropout/Shape?
=sequential_23/dropout_24/dropout/random_uniform/RandomUniformRandomUniform/sequential_23/dropout_24/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02?
=sequential_23/dropout_24/dropout/random_uniform/RandomUniform?
/sequential_23/dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>21
/sequential_23/dropout_24/dropout/GreaterEqual/y?
-sequential_23/dropout_24/dropout/GreaterEqualGreaterEqualFsequential_23/dropout_24/dropout/random_uniform/RandomUniform:output:08sequential_23/dropout_24/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2/
-sequential_23/dropout_24/dropout/GreaterEqual?
%sequential_23/dropout_24/dropout/CastCast1sequential_23/dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2'
%sequential_23/dropout_24/dropout/Cast?
&sequential_23/dropout_24/dropout/Mul_1Mul(sequential_23/dropout_24/dropout/Mul:z:0)sequential_23/dropout_24/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2(
&sequential_23/dropout_24/dropout/Mul_1?
@sequential_23/global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2B
@sequential_23/global_average_pooling1d_12/Mean/reduction_indices?
.sequential_23/global_average_pooling1d_12/MeanMean*sequential_23/dropout_24/dropout/Mul_1:z:0Isequential_23/global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????20
.sequential_23/global_average_pooling1d_12/Mean?
&sequential_23/dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&sequential_23/dropout_25/dropout/Const?
$sequential_23/dropout_25/dropout/MulMul7sequential_23/global_average_pooling1d_12/Mean:output:0/sequential_23/dropout_25/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2&
$sequential_23/dropout_25/dropout/Mul?
&sequential_23/dropout_25/dropout/ShapeShape7sequential_23/global_average_pooling1d_12/Mean:output:0*
T0*
_output_shapes
:2(
&sequential_23/dropout_25/dropout/Shape?
=sequential_23/dropout_25/dropout/random_uniform/RandomUniformRandomUniform/sequential_23/dropout_25/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02?
=sequential_23/dropout_25/dropout/random_uniform/RandomUniform?
/sequential_23/dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>21
/sequential_23/dropout_25/dropout/GreaterEqual/y?
-sequential_23/dropout_25/dropout/GreaterEqualGreaterEqualFsequential_23/dropout_25/dropout/random_uniform/RandomUniform:output:08sequential_23/dropout_25/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2/
-sequential_23/dropout_25/dropout/GreaterEqual?
%sequential_23/dropout_25/dropout/CastCast1sequential_23/dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2'
%sequential_23/dropout_25/dropout/Cast?
&sequential_23/dropout_25/dropout/Mul_1Mul(sequential_23/dropout_25/dropout/Mul:z:0)sequential_23/dropout_25/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2(
&sequential_23/dropout_25/dropout/Mul_1?
,sequential_23/dense_12/MatMul/ReadVariableOpReadVariableOp5sequential_23_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_23/dense_12/MatMul/ReadVariableOp?
sequential_23/dense_12/MatMulMatMul*sequential_23/dropout_25/dropout/Mul_1:z:04sequential_23/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_23/dense_12/MatMul?
-sequential_23/dense_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_23_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_23/dense_12/BiasAdd/ReadVariableOp?
sequential_23/dense_12/BiasAddBiasAdd'sequential_23/dense_12/MatMul:product:05sequential_23/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_23/dense_12/BiasAdd?
activation_11/SigmoidSigmoid'sequential_23/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
activation_11/Sigmoid?
IdentityIdentityactivation_11/Sigmoid:y:0.^sequential_23/dense_12/BiasAdd/ReadVariableOp-^sequential_23/dense_12/MatMul/ReadVariableOp,^sequential_23/embedding_12/embedding_lookupP^text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::2^
-sequential_23/dense_12/BiasAdd/ReadVariableOp-sequential_23/dense_12/BiasAdd/ReadVariableOp2\
,sequential_23/dense_12/MatMul/ReadVariableOp,sequential_23/dense_12/MatMul/ReadVariableOp2Z
+sequential_23/embedding_12/embedding_lookup+sequential_23/embedding_12/embedding_lookup2?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
.__inference_sequential_24_layer_call_fn_310749

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_3104522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
d
F__inference_dropout_24_layer_call_and_return_conditional_losses_310955

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
'text_vectorization_12_cond_false_310568*
&text_vectorization_12_cond_placeholderf
btext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
.text_vectorization_12/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.text_vectorization_12/cond/strided_slice/stack?
0text_vectorization_12/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   22
0text_vectorization_12/cond/strided_slice/stack_1?
0text_vectorization_12/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0text_vectorization_12/cond/strided_slice/stack_2?
(text_vectorization_12/cond/strided_sliceStridedSlicebtext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor7text_vectorization_12/cond/strided_slice/stack:output:09text_vectorization_12/cond/strided_slice/stack_1:output:09text_vectorization_12/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2*
(text_vectorization_12/cond/strided_slice?
#text_vectorization_12/cond/IdentityIdentity1text_vectorization_12/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
&text_vectorization_12_cond_true_310050G
Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub\
Xtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
+text_vectorization_12/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2-
+text_vectorization_12/cond/Pad/paddings/1/0?
)text_vectorization_12/cond/Pad/paddings/1Pack4text_vectorization_12/cond/Pad/paddings/1/0:output:0Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub*
N*
T0*
_output_shapes
:2+
)text_vectorization_12/cond/Pad/paddings/1?
+text_vectorization_12/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+text_vectorization_12/cond/Pad/paddings/0_1?
'text_vectorization_12/cond/Pad/paddingsPack4text_vectorization_12/cond/Pad/paddings/0_1:output:02text_vectorization_12/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2)
'text_vectorization_12/cond/Pad/paddings?
text_vectorization_12/cond/PadPadXtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor0text_vectorization_12/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2 
text_vectorization_12/cond/Pad?
#text_vectorization_12/cond/IdentityIdentity'text_vectorization_12/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
+
__inference_<lambda>_311080
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
e
I__inference_activation_11_layer_call_and_return_conditional_losses_310916

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_25_layer_call_fn_311014

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_3098832
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_23_layer_call_and_return_conditional_losses_309982

inputs
embedding_12_309970
dense_12_309976
dense_12_309978
identity?? dense_12/StatefulPartitionedCall?$embedding_12/StatefulPartitionedCall?
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_12_309970*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_3098122&
$embedding_12/StatefulPartitionedCall?
dropout_24/PartitionedCallPartitionedCall-embedding_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_24_layer_call_and_return_conditional_losses_3098412
dropout_24/PartitionedCall?
+global_average_pooling1d_12/PartitionedCallPartitionedCall#dropout_24/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_3098592-
+global_average_pooling1d_12/PartitionedCall?
dropout_25/PartitionedCallPartitionedCall4global_average_pooling1d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_3098832
dropout_25/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dense_12_309976dense_12_309978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3099062"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_24_layer_call_fn_310465
text_vectorization_12_input
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_3104522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::22
StatefulPartitionedCallStatefulPartitionedCall:d `
'
_output_shapes
:?????????
5
_user_specified_nametext_vectorization_12_input:

_output_shapes
: 
?	
?
D__inference_dense_12_layer_call_and_return_conditional_losses_311024

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_23_layer_call_and_return_conditional_losses_310813

inputs	(
$embedding_12_embedding_lookup_310797+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?embedding_12/embedding_lookup^
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????2
Castz
embedding_12/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:??????????2
embedding_12/Cast?
embedding_12/embedding_lookupResourceGather$embedding_12_embedding_lookup_310797embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_12/embedding_lookup/310797*,
_output_shapes
:??????????*
dtype02
embedding_12/embedding_lookup?
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_12/embedding_lookup/310797*,
_output_shapes
:??????????2(
&embedding_12/embedding_lookup/Identity?
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_12/embedding_lookup/Identity_1?
dropout_24/IdentityIdentity1embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
dropout_24/Identity?
2global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2global_average_pooling1d_12/Mean/reduction_indices?
 global_average_pooling1d_12/MeanMeandropout_24/Identity:output:0;global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2"
 global_average_pooling1d_12/Mean?
dropout_25/IdentityIdentity)global_average_pooling1d_12/Mean:output:0*
T0*'
_output_shapes
:?????????2
dropout_25/Identity?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_25/Identity:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd?
IdentityIdentitydense_12/BiasAdd:output:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp^embedding_12/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????:::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2>
embedding_12/embedding_lookupembedding_12/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
X
<__inference_global_average_pooling1d_12_layer_call_fn_310987

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_3097952
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_12_layer_call_and_return_conditional_losses_309906

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&text_vectorization_12_cond_true_310567G
Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub\
Xtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
+text_vectorization_12/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2-
+text_vectorization_12/cond/Pad/paddings/1/0?
)text_vectorization_12/cond/Pad/paddings/1Pack4text_vectorization_12/cond/Pad/paddings/1/0:output:0Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub*
N*
T0*
_output_shapes
:2+
)text_vectorization_12/cond/Pad/paddings/1?
+text_vectorization_12/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+text_vectorization_12/cond/Pad/paddings/0_1?
'text_vectorization_12/cond/Pad/paddingsPack4text_vectorization_12/cond/Pad/paddings/0_1:output:02text_vectorization_12/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2)
'text_vectorization_12/cond/Pad/paddings?
text_vectorization_12/cond/PadPadXtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor0text_vectorization_12/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2 
text_vectorization_12/cond/Pad?
#text_vectorization_12/cond/IdentityIdentity'text_vectorization_12/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
&text_vectorization_12_cond_true_310228G
Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub\
Xtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
+text_vectorization_12/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2-
+text_vectorization_12/cond/Pad/paddings/1/0?
)text_vectorization_12/cond/Pad/paddings/1Pack4text_vectorization_12/cond/Pad/paddings/1/0:output:0Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub*
N*
T0*
_output_shapes
:2+
)text_vectorization_12/cond/Pad/paddings/1?
+text_vectorization_12/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+text_vectorization_12/cond/Pad/paddings/0_1?
'text_vectorization_12/cond/Pad/paddingsPack4text_vectorization_12/cond/Pad/paddings/0_1:output:02text_vectorization_12/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2)
'text_vectorization_12/cond/Pad/paddings?
text_vectorization_12/cond/PadPadXtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor0text_vectorization_12/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2 
text_vectorization_12/cond/Pad?
#text_vectorization_12/cond/IdentityIdentity'text_vectorization_12/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
Ѿ
?
!__inference__wrapped_model_309779
text_vectorization_12_inputn
jsequential_24_text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handleo
ksequential_24_text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value	D
@sequential_24_sequential_23_embedding_12_embedding_lookup_309762G
Csequential_24_sequential_23_dense_12_matmul_readvariableop_resourceH
Dsequential_24_sequential_23_dense_12_biasadd_readvariableop_resource
identity??;sequential_24/sequential_23/dense_12/BiasAdd/ReadVariableOp?:sequential_24/sequential_23/dense_12/MatMul/ReadVariableOp?9sequential_24/sequential_23/embedding_12/embedding_lookup?]sequential_24/text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
/sequential_24/text_vectorization_12/StringLowerStringLowertext_vectorization_12_input*'
_output_shapes
:?????????21
/sequential_24/text_vectorization_12/StringLower?
6sequential_24/text_vectorization_12/StaticRegexReplaceStaticRegexReplace8sequential_24/text_vectorization_12/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 28
6sequential_24/text_vectorization_12/StaticRegexReplace?
8sequential_24/text_vectorization_12/StaticRegexReplace_1StaticRegexReplace?sequential_24/text_vectorization_12/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2:
8sequential_24/text_vectorization_12/StaticRegexReplace_1?
+sequential_24/text_vectorization_12/SqueezeSqueezeAsequential_24/text_vectorization_12/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2-
+sequential_24/text_vectorization_12/Squeeze?
5sequential_24/text_vectorization_12/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 27
5sequential_24/text_vectorization_12/StringSplit/Const?
=sequential_24/text_vectorization_12/StringSplit/StringSplitV2StringSplitV24sequential_24/text_vectorization_12/Squeeze:output:0>sequential_24/text_vectorization_12/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2?
=sequential_24/text_vectorization_12/StringSplit/StringSplitV2?
Csequential_24/text_vectorization_12/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2E
Csequential_24/text_vectorization_12/StringSplit/strided_slice/stack?
Esequential_24/text_vectorization_12/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2G
Esequential_24/text_vectorization_12/StringSplit/strided_slice/stack_1?
Esequential_24/text_vectorization_12/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2G
Esequential_24/text_vectorization_12/StringSplit/strided_slice/stack_2?
=sequential_24/text_vectorization_12/StringSplit/strided_sliceStridedSliceGsequential_24/text_vectorization_12/StringSplit/StringSplitV2:indices:0Lsequential_24/text_vectorization_12/StringSplit/strided_slice/stack:output:0Nsequential_24/text_vectorization_12/StringSplit/strided_slice/stack_1:output:0Nsequential_24/text_vectorization_12/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2?
=sequential_24/text_vectorization_12/StringSplit/strided_slice?
Esequential_24/text_vectorization_12/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_24/text_vectorization_12/StringSplit/strided_slice_1/stack?
Gsequential_24/text_vectorization_12/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_24/text_vectorization_12/StringSplit/strided_slice_1/stack_1?
Gsequential_24/text_vectorization_12/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_24/text_vectorization_12/StringSplit/strided_slice_1/stack_2?
?sequential_24/text_vectorization_12/StringSplit/strided_slice_1StridedSliceEsequential_24/text_vectorization_12/StringSplit/StringSplitV2:shape:0Nsequential_24/text_vectorization_12/StringSplit/strided_slice_1/stack:output:0Psequential_24/text_vectorization_12/StringSplit/strided_slice_1/stack_1:output:0Psequential_24/text_vectorization_12/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2A
?sequential_24/text_vectorization_12/StringSplit/strided_slice_1?
fsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastFsequential_24/text_vectorization_12/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2h
fsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
hsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastHsequential_24/text_vectorization_12/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2j
hsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
psequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapejsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2r
psequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
psequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2r
psequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
osequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdysequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ysequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2q
osequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
tsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2v
tsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterxsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0}sequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2t
rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
osequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastvsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2q
osequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2t
rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
nsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxjsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0{sequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2p
nsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
psequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2r
psequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
nsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2wsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ysequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2p
nsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
nsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulssequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2p
nsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumlsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2t
rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumlsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0vsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2t
rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2t
rsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
ssequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountjsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0vsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0{sequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2u
ssequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
msequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2o
msequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
hsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumzsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0vsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2j
hsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
qsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2s
qsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
msequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2o
msequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
hsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2zsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0nsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0vsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2j
hsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
]sequential_24/text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2LookupTableFindV2jsequential_24_text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handleFsequential_24/text_vectorization_12/StringSplit/StringSplitV2:values:0ksequential_24_text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2_
]sequential_24/text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
Fsequential_24/text_vectorization_12/string_lookup_12/assert_equal/NoOpNoOp*
_output_shapes
 2H
Fsequential_24/text_vectorization_12/string_lookup_12/assert_equal/NoOp?
=sequential_24/text_vectorization_12/string_lookup_12/IdentityIdentityfsequential_24/text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2?
=sequential_24/text_vectorization_12/string_lookup_12/Identity?
?sequential_24/text_vectorization_12/string_lookup_12/Identity_1Identityqsequential_24/text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????2A
?sequential_24/text_vectorization_12/string_lookup_12/Identity_1?
@sequential_24/text_vectorization_12/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2B
@sequential_24/text_vectorization_12/RaggedToTensor/default_value?
8sequential_24/text_vectorization_12/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2:
8sequential_24/text_vectorization_12/RaggedToTensor/Const?
Gsequential_24/text_vectorization_12/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorAsequential_24/text_vectorization_12/RaggedToTensor/Const:output:0Fsequential_24/text_vectorization_12/string_lookup_12/Identity:output:0Isequential_24/text_vectorization_12/RaggedToTensor/default_value:output:0Hsequential_24/text_vectorization_12/string_lookup_12/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2I
Gsequential_24/text_vectorization_12/RaggedToTensor/RaggedTensorToTensor?
)sequential_24/text_vectorization_12/ShapeShapePsequential_24/text_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2+
)sequential_24/text_vectorization_12/Shape?
7sequential_24/text_vectorization_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7sequential_24/text_vectorization_12/strided_slice/stack?
9sequential_24/text_vectorization_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_24/text_vectorization_12/strided_slice/stack_1?
9sequential_24/text_vectorization_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_24/text_vectorization_12/strided_slice/stack_2?
1sequential_24/text_vectorization_12/strided_sliceStridedSlice2sequential_24/text_vectorization_12/Shape:output:0@sequential_24/text_vectorization_12/strided_slice/stack:output:0Bsequential_24/text_vectorization_12/strided_slice/stack_1:output:0Bsequential_24/text_vectorization_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_24/text_vectorization_12/strided_slice?
)sequential_24/text_vectorization_12/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2+
)sequential_24/text_vectorization_12/sub/x?
'sequential_24/text_vectorization_12/subSub2sequential_24/text_vectorization_12/sub/x:output:0:sequential_24/text_vectorization_12/strided_slice:output:0*
T0*
_output_shapes
: 2)
'sequential_24/text_vectorization_12/sub?
*sequential_24/text_vectorization_12/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2,
*sequential_24/text_vectorization_12/Less/y?
(sequential_24/text_vectorization_12/LessLess:sequential_24/text_vectorization_12/strided_slice:output:03sequential_24/text_vectorization_12/Less/y:output:0*
T0*
_output_shapes
: 2*
(sequential_24/text_vectorization_12/Less?
(sequential_24/text_vectorization_12/condStatelessIf,sequential_24/text_vectorization_12/Less:z:0+sequential_24/text_vectorization_12/sub:z:0Psequential_24/text_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *H
else_branch9R7
5sequential_24_text_vectorization_12_cond_false_309740*/
output_shapes
:??????????????????*G
then_branch8R6
4sequential_24_text_vectorization_12_cond_true_3097392*
(sequential_24/text_vectorization_12/cond?
1sequential_24/text_vectorization_12/cond/IdentityIdentity1sequential_24/text_vectorization_12/cond:output:0*
T0	*(
_output_shapes
:??????????23
1sequential_24/text_vectorization_12/cond/Identity?
 sequential_24/sequential_23/CastCast:sequential_24/text_vectorization_12/cond/Identity:output:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????2"
 sequential_24/sequential_23/Cast?
-sequential_24/sequential_23/embedding_12/CastCast$sequential_24/sequential_23/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:??????????2/
-sequential_24/sequential_23/embedding_12/Cast?
9sequential_24/sequential_23/embedding_12/embedding_lookupResourceGather@sequential_24_sequential_23_embedding_12_embedding_lookup_3097621sequential_24/sequential_23/embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*S
_classI
GEloc:@sequential_24/sequential_23/embedding_12/embedding_lookup/309762*,
_output_shapes
:??????????*
dtype02;
9sequential_24/sequential_23/embedding_12/embedding_lookup?
Bsequential_24/sequential_23/embedding_12/embedding_lookup/IdentityIdentityBsequential_24/sequential_23/embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*S
_classI
GEloc:@sequential_24/sequential_23/embedding_12/embedding_lookup/309762*,
_output_shapes
:??????????2D
Bsequential_24/sequential_23/embedding_12/embedding_lookup/Identity?
Dsequential_24/sequential_23/embedding_12/embedding_lookup/Identity_1IdentityKsequential_24/sequential_23/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2F
Dsequential_24/sequential_23/embedding_12/embedding_lookup/Identity_1?
/sequential_24/sequential_23/dropout_24/IdentityIdentityMsequential_24/sequential_23/embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????21
/sequential_24/sequential_23/dropout_24/Identity?
Nsequential_24/sequential_23/global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2P
Nsequential_24/sequential_23/global_average_pooling1d_12/Mean/reduction_indices?
<sequential_24/sequential_23/global_average_pooling1d_12/MeanMean8sequential_24/sequential_23/dropout_24/Identity:output:0Wsequential_24/sequential_23/global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2>
<sequential_24/sequential_23/global_average_pooling1d_12/Mean?
/sequential_24/sequential_23/dropout_25/IdentityIdentityEsequential_24/sequential_23/global_average_pooling1d_12/Mean:output:0*
T0*'
_output_shapes
:?????????21
/sequential_24/sequential_23/dropout_25/Identity?
:sequential_24/sequential_23/dense_12/MatMul/ReadVariableOpReadVariableOpCsequential_24_sequential_23_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:sequential_24/sequential_23/dense_12/MatMul/ReadVariableOp?
+sequential_24/sequential_23/dense_12/MatMulMatMul8sequential_24/sequential_23/dropout_25/Identity:output:0Bsequential_24/sequential_23/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+sequential_24/sequential_23/dense_12/MatMul?
;sequential_24/sequential_23/dense_12/BiasAdd/ReadVariableOpReadVariableOpDsequential_24_sequential_23_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_24/sequential_23/dense_12/BiasAdd/ReadVariableOp?
,sequential_24/sequential_23/dense_12/BiasAddBiasAdd5sequential_24/sequential_23/dense_12/MatMul:product:0Csequential_24/sequential_23/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,sequential_24/sequential_23/dense_12/BiasAdd?
#sequential_24/activation_11/SigmoidSigmoid5sequential_24/sequential_23/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2%
#sequential_24/activation_11/Sigmoid?
IdentityIdentity'sequential_24/activation_11/Sigmoid:y:0<^sequential_24/sequential_23/dense_12/BiasAdd/ReadVariableOp;^sequential_24/sequential_23/dense_12/MatMul/ReadVariableOp:^sequential_24/sequential_23/embedding_12/embedding_lookup^^sequential_24/text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::2z
;sequential_24/sequential_23/dense_12/BiasAdd/ReadVariableOp;sequential_24/sequential_23/dense_12/BiasAdd/ReadVariableOp2x
:sequential_24/sequential_23/dense_12/MatMul/ReadVariableOp:sequential_24/sequential_23/dense_12/MatMul/ReadVariableOp2v
9sequential_24/sequential_23/embedding_12/embedding_lookup9sequential_24/sequential_23/embedding_12/embedding_lookup2?
]sequential_24/text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2]sequential_24/text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:d `
'
_output_shapes
:?????????
5
_user_specified_nametext_vectorization_12_input:

_output_shapes
: 
?
?
&text_vectorization_12_cond_true_310319G
Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub\
Xtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
+text_vectorization_12/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2-
+text_vectorization_12/cond/Pad/paddings/1/0?
)text_vectorization_12/cond/Pad/paddings/1Pack4text_vectorization_12/cond/Pad/paddings/1/0:output:0Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub*
N*
T0*
_output_shapes
:2+
)text_vectorization_12/cond/Pad/paddings/1?
+text_vectorization_12/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+text_vectorization_12/cond/Pad/paddings/0_1?
'text_vectorization_12/cond/Pad/paddingsPack4text_vectorization_12/cond/Pad/paddings/0_1:output:02text_vectorization_12/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2)
'text_vectorization_12/cond/Pad/paddings?
text_vectorization_12/cond/PadPadXtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor0text_vectorization_12/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2 
text_vectorization_12/cond/Pad?
#text_vectorization_12/cond/IdentityIdentity'text_vectorization_12/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
ގ
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_310258
text_vectorization_12_input`
\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handlea
]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value	
sequential_23_310249
sequential_23_310251
sequential_23_310253
identity??%sequential_23/StatefulPartitionedCall?Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
!text_vectorization_12/StringLowerStringLowertext_vectorization_12_input*'
_output_shapes
:?????????2#
!text_vectorization_12/StringLower?
(text_vectorization_12/StaticRegexReplaceStaticRegexReplace*text_vectorization_12/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2*
(text_vectorization_12/StaticRegexReplace?
*text_vectorization_12/StaticRegexReplace_1StaticRegexReplace1text_vectorization_12/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2,
*text_vectorization_12/StaticRegexReplace_1?
text_vectorization_12/SqueezeSqueeze3text_vectorization_12/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_12/Squeeze?
'text_vectorization_12/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2)
'text_vectorization_12/StringSplit/Const?
/text_vectorization_12/StringSplit/StringSplitV2StringSplitV2&text_vectorization_12/Squeeze:output:00text_vectorization_12/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:21
/text_vectorization_12/StringSplit/StringSplitV2?
5text_vectorization_12/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5text_vectorization_12/StringSplit/strided_slice/stack?
7text_vectorization_12/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7text_vectorization_12/StringSplit/strided_slice/stack_1?
7text_vectorization_12/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7text_vectorization_12/StringSplit/strided_slice/stack_2?
/text_vectorization_12/StringSplit/strided_sliceStridedSlice9text_vectorization_12/StringSplit/StringSplitV2:indices:0>text_vectorization_12/StringSplit/strided_slice/stack:output:0@text_vectorization_12/StringSplit/strided_slice/stack_1:output:0@text_vectorization_12/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask21
/text_vectorization_12/StringSplit/strided_slice?
7text_vectorization_12/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7text_vectorization_12/StringSplit/strided_slice_1/stack?
9text_vectorization_12/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_1?
9text_vectorization_12/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_2?
1text_vectorization_12/StringSplit/strided_slice_1StridedSlice7text_vectorization_12/StringSplit/StringSplitV2:shape:0@text_vectorization_12/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask23
1text_vectorization_12/StringSplit/strided_slice_1?
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_12/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Z
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_12/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2h
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2g
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2e
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2LookupTableFindV2\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handle8text_vectorization_12/StringSplit/StringSplitV2:values:0]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2Q
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
8text_vectorization_12/string_lookup_12/assert_equal/NoOpNoOp*
_output_shapes
 2:
8text_vectorization_12/string_lookup_12/assert_equal/NoOp?
/text_vectorization_12/string_lookup_12/IdentityIdentityXtext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_12/string_lookup_12/Identity?
1text_vectorization_12/string_lookup_12/Identity_1Identityctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????23
1text_vectorization_12/string_lookup_12/Identity_1?
2text_vectorization_12/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 24
2text_vectorization_12/RaggedToTensor/default_value?
*text_vectorization_12/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2,
*text_vectorization_12/RaggedToTensor/Const?
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_12/RaggedToTensor/Const:output:08text_vectorization_12/string_lookup_12/Identity:output:0;text_vectorization_12/RaggedToTensor/default_value:output:0:text_vectorization_12/string_lookup_12/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2;
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_12/ShapeShapeBtext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_12/Shape?
)text_vectorization_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)text_vectorization_12/strided_slice/stack?
+text_vectorization_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_1?
+text_vectorization_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_2?
#text_vectorization_12/strided_sliceStridedSlice$text_vectorization_12/Shape:output:02text_vectorization_12/strided_slice/stack:output:04text_vectorization_12/strided_slice/stack_1:output:04text_vectorization_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#text_vectorization_12/strided_slice}
text_vectorization_12/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/sub/x?
text_vectorization_12/subSub$text_vectorization_12/sub/x:output:0,text_vectorization_12/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/sub
text_vectorization_12/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/Less/y?
text_vectorization_12/LessLess,text_vectorization_12/strided_slice:output:0%text_vectorization_12/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/Less?
text_vectorization_12/condStatelessIftext_vectorization_12/Less:z:0text_vectorization_12/sub:z:0Btext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'text_vectorization_12_cond_false_310229*/
output_shapes
:??????????????????*9
then_branch*R(
&text_vectorization_12_cond_true_3102282
text_vectorization_12/cond?
#text_vectorization_12/cond/IdentityIdentity#text_vectorization_12/cond:output:0*
T0	*(
_output_shapes
:??????????2%
#text_vectorization_12/cond/Identity?
%sequential_23/StatefulPartitionedCallStatefulPartitionedCall,text_vectorization_12/cond/Identity:output:0sequential_23_310249sequential_23_310251sequential_23_310253*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_3101262'
%sequential_23/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall.sequential_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_11_layer_call_and_return_conditional_losses_3101612
activation_11/PartitionedCall?
IdentityIdentity&activation_11/PartitionedCall:output:0&^sequential_23/StatefulPartitionedCallP^text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::2N
%sequential_23/StatefulPartitionedCall%sequential_23/StatefulPartitionedCall2?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:d `
'
_output_shapes
:?????????
5
_user_specified_nametext_vectorization_12_input:

_output_shapes
: 
??
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_310349

inputs`
\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handlea
]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value	
sequential_23_310340
sequential_23_310342
sequential_23_310344
identity??%sequential_23/StatefulPartitionedCall?Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
!text_vectorization_12/StringLowerStringLowerinputs*'
_output_shapes
:?????????2#
!text_vectorization_12/StringLower?
(text_vectorization_12/StaticRegexReplaceStaticRegexReplace*text_vectorization_12/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2*
(text_vectorization_12/StaticRegexReplace?
*text_vectorization_12/StaticRegexReplace_1StaticRegexReplace1text_vectorization_12/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2,
*text_vectorization_12/StaticRegexReplace_1?
text_vectorization_12/SqueezeSqueeze3text_vectorization_12/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_12/Squeeze?
'text_vectorization_12/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2)
'text_vectorization_12/StringSplit/Const?
/text_vectorization_12/StringSplit/StringSplitV2StringSplitV2&text_vectorization_12/Squeeze:output:00text_vectorization_12/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:21
/text_vectorization_12/StringSplit/StringSplitV2?
5text_vectorization_12/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5text_vectorization_12/StringSplit/strided_slice/stack?
7text_vectorization_12/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7text_vectorization_12/StringSplit/strided_slice/stack_1?
7text_vectorization_12/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7text_vectorization_12/StringSplit/strided_slice/stack_2?
/text_vectorization_12/StringSplit/strided_sliceStridedSlice9text_vectorization_12/StringSplit/StringSplitV2:indices:0>text_vectorization_12/StringSplit/strided_slice/stack:output:0@text_vectorization_12/StringSplit/strided_slice/stack_1:output:0@text_vectorization_12/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask21
/text_vectorization_12/StringSplit/strided_slice?
7text_vectorization_12/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7text_vectorization_12/StringSplit/strided_slice_1/stack?
9text_vectorization_12/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_1?
9text_vectorization_12/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_2?
1text_vectorization_12/StringSplit/strided_slice_1StridedSlice7text_vectorization_12/StringSplit/StringSplitV2:shape:0@text_vectorization_12/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask23
1text_vectorization_12/StringSplit/strided_slice_1?
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_12/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Z
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_12/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2h
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2g
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2e
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2LookupTableFindV2\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handle8text_vectorization_12/StringSplit/StringSplitV2:values:0]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2Q
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
8text_vectorization_12/string_lookup_12/assert_equal/NoOpNoOp*
_output_shapes
 2:
8text_vectorization_12/string_lookup_12/assert_equal/NoOp?
/text_vectorization_12/string_lookup_12/IdentityIdentityXtext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_12/string_lookup_12/Identity?
1text_vectorization_12/string_lookup_12/Identity_1Identityctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????23
1text_vectorization_12/string_lookup_12/Identity_1?
2text_vectorization_12/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 24
2text_vectorization_12/RaggedToTensor/default_value?
*text_vectorization_12/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2,
*text_vectorization_12/RaggedToTensor/Const?
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_12/RaggedToTensor/Const:output:08text_vectorization_12/string_lookup_12/Identity:output:0;text_vectorization_12/RaggedToTensor/default_value:output:0:text_vectorization_12/string_lookup_12/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2;
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_12/ShapeShapeBtext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_12/Shape?
)text_vectorization_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)text_vectorization_12/strided_slice/stack?
+text_vectorization_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_1?
+text_vectorization_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_2?
#text_vectorization_12/strided_sliceStridedSlice$text_vectorization_12/Shape:output:02text_vectorization_12/strided_slice/stack:output:04text_vectorization_12/strided_slice/stack_1:output:04text_vectorization_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#text_vectorization_12/strided_slice}
text_vectorization_12/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/sub/x?
text_vectorization_12/subSub$text_vectorization_12/sub/x:output:0,text_vectorization_12/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/sub
text_vectorization_12/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/Less/y?
text_vectorization_12/LessLess,text_vectorization_12/strided_slice:output:0%text_vectorization_12/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/Less?
text_vectorization_12/condStatelessIftext_vectorization_12/Less:z:0text_vectorization_12/sub:z:0Btext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'text_vectorization_12_cond_false_310320*/
output_shapes
:??????????????????*9
then_branch*R(
&text_vectorization_12_cond_true_3103192
text_vectorization_12/cond?
#text_vectorization_12/cond/IdentityIdentity#text_vectorization_12/cond:output:0*
T0	*(
_output_shapes
:??????????2%
#text_vectorization_12/cond/Identity?
%sequential_23/StatefulPartitionedCallStatefulPartitionedCall,text_vectorization_12/cond/Identity:output:0sequential_23_310340sequential_23_310342sequential_23_310344*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_3101052'
%sequential_23/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall.sequential_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_11_layer_call_and_return_conditional_losses_3101612
activation_11/PartitionedCall?
IdentityIdentity&activation_11/PartitionedCall:output:0&^sequential_23/StatefulPartitionedCallP^text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::2N
%sequential_23/StatefulPartitionedCall%sequential_23/StatefulPartitionedCall2?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
s
-__inference_embedding_12_layer_call_fn_310938

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_3098122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_24_layer_call_fn_310965

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_24_layer_call_and_return_conditional_losses_3098412
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
'text_vectorization_12_cond_false_310423*
&text_vectorization_12_cond_placeholderf
btext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
.text_vectorization_12/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.text_vectorization_12/cond/strided_slice/stack?
0text_vectorization_12/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   22
0text_vectorization_12/cond/strided_slice/stack_1?
0text_vectorization_12/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0text_vectorization_12/cond/strided_slice/stack_2?
(text_vectorization_12/cond/strided_sliceStridedSlicebtext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor7text_vectorization_12/cond/strided_slice/stack:output:09text_vectorization_12/cond/strided_slice/stack_1:output:09text_vectorization_12/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2*
(text_vectorization_12/cond/strided_slice?
#text_vectorization_12/cond/IdentityIdentity1text_vectorization_12/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
X
<__inference_global_average_pooling1d_12_layer_call_fn_310976

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_3098592
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?-
?
I__inference_sequential_23_layer_call_and_return_conditional_losses_310869

inputs(
$embedding_12_embedding_lookup_310839+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?embedding_12/embedding_lookup?
embedding_12/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????2
embedding_12/Cast?
embedding_12/embedding_lookupResourceGather$embedding_12_embedding_lookup_310839embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_12/embedding_lookup/310839*4
_output_shapes"
 :??????????????????*
dtype02
embedding_12/embedding_lookup?
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_12/embedding_lookup/310839*4
_output_shapes"
 :??????????????????2(
&embedding_12/embedding_lookup/Identity?
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????2*
(embedding_12/embedding_lookup/Identity_1y
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_24/dropout/Const?
dropout_24/dropout/MulMul1embedding_12/embedding_lookup/Identity_1:output:0!dropout_24/dropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout_24/dropout/Mul?
dropout_24/dropout/ShapeShape1embedding_12/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dropout_24/dropout/Shape?
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype021
/dropout_24/dropout/random_uniform/RandomUniform?
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_24/dropout/GreaterEqual/y?
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2!
dropout_24/dropout/GreaterEqual?
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout_24/dropout/Cast?
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
dropout_24/dropout/Mul_1?
2global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2global_average_pooling1d_12/Mean/reduction_indices?
 global_average_pooling1d_12/MeanMeandropout_24/dropout/Mul_1:z:0;global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2"
 global_average_pooling1d_12/Meany
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_25/dropout/Const?
dropout_25/dropout/MulMul)global_average_pooling1d_12/Mean:output:0!dropout_25/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_25/dropout/Mul?
dropout_25/dropout/ShapeShape)global_average_pooling1d_12/Mean:output:0*
T0*
_output_shapes
:2
dropout_25/dropout/Shape?
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype021
/dropout_25/dropout/random_uniform/RandomUniform?
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_25/dropout/GreaterEqual/y?
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2!
dropout_25/dropout/GreaterEqual?
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_25/dropout/Cast?
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_25/dropout/Mul_1?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_25/dropout/Mul_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd?
IdentityIdentitydense_12/BiasAdd:output:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp^embedding_12/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2>
embedding_12/embedding_lookupembedding_12/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?-
?
I__inference_sequential_23_layer_call_and_return_conditional_losses_310105

inputs	(
$embedding_12_embedding_lookup_310075+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?embedding_12/embedding_lookup^
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????2
Castz
embedding_12/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:??????????2
embedding_12/Cast?
embedding_12/embedding_lookupResourceGather$embedding_12_embedding_lookup_310075embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_12/embedding_lookup/310075*,
_output_shapes
:??????????*
dtype02
embedding_12/embedding_lookup?
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_12/embedding_lookup/310075*,
_output_shapes
:??????????2(
&embedding_12/embedding_lookup/Identity?
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_12/embedding_lookup/Identity_1y
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_24/dropout/Const?
dropout_24/dropout/MulMul1embedding_12/embedding_lookup/Identity_1:output:0!dropout_24/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_24/dropout/Mul?
dropout_24/dropout/ShapeShape1embedding_12/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dropout_24/dropout/Shape?
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_24/dropout/random_uniform/RandomUniform?
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_24/dropout/GreaterEqual/y?
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_24/dropout/GreaterEqual?
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_24/dropout/Cast?
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_24/dropout/Mul_1?
2global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2global_average_pooling1d_12/Mean/reduction_indices?
 global_average_pooling1d_12/MeanMeandropout_24/dropout/Mul_1:z:0;global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2"
 global_average_pooling1d_12/Meany
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_25/dropout/Const?
dropout_25/dropout/MulMul)global_average_pooling1d_12/Mean:output:0!dropout_25/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_25/dropout/Mul?
dropout_25/dropout/ShapeShape)global_average_pooling1d_12/Mean:output:0*
T0*
_output_shapes
:2
dropout_25/dropout/Shape?
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype021
/dropout_25/dropout/random_uniform/RandomUniform?
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_25/dropout/GreaterEqual/y?
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2!
dropout_25/dropout/GreaterEqual?
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_25/dropout/Cast?
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_25/dropout/Mul_1?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_25/dropout/Mul_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd?
IdentityIdentitydense_12/BiasAdd:output:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp^embedding_12/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????:::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2>
embedding_12/embedding_lookupembedding_12/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&text_vectorization_12_cond_true_310422G
Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub\
Xtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
+text_vectorization_12/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2-
+text_vectorization_12/cond/Pad/paddings/1/0?
)text_vectorization_12/cond/Pad/paddings/1Pack4text_vectorization_12/cond/Pad/paddings/1/0:output:0Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub*
N*
T0*
_output_shapes
:2+
)text_vectorization_12/cond/Pad/paddings/1?
+text_vectorization_12/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+text_vectorization_12/cond/Pad/paddings/0_1?
'text_vectorization_12/cond/Pad/paddingsPack4text_vectorization_12/cond/Pad/paddings/0_1:output:02text_vectorization_12/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2)
'text_vectorization_12/cond/Pad/paddings?
text_vectorization_12/cond/PadPadXtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor0text_vectorization_12/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2 
text_vectorization_12/cond/Pad?
#text_vectorization_12/cond/IdentityIdentity'text_vectorization_12/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
I__inference_sequential_23_layer_call_and_return_conditional_losses_309956

inputs
embedding_12_309944
dense_12_309950
dense_12_309952
identity?? dense_12/StatefulPartitionedCall?"dropout_24/StatefulPartitionedCall?"dropout_25/StatefulPartitionedCall?$embedding_12/StatefulPartitionedCall?
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_12_309944*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_3098122&
$embedding_12/StatefulPartitionedCall?
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall-embedding_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_24_layer_call_and_return_conditional_losses_3098362$
"dropout_24/StatefulPartitionedCall?
+global_average_pooling1d_12/PartitionedCallPartitionedCall+dropout_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_3098592-
+global_average_pooling1d_12/PartitionedCall?
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_12/PartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_3098782$
"dropout_25/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dense_12_309950dense_12_309952*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3099062"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_24_layer_call_and_return_conditional_losses_310950

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
5sequential_24_text_vectorization_12_cond_false_3097408
4sequential_24_text_vectorization_12_cond_placeholder?
~sequential_24_text_vectorization_12_cond_strided_slice_sequential_24_text_vectorization_12_raggedtotensor_raggedtensortotensor	5
1sequential_24_text_vectorization_12_cond_identity	?
<sequential_24/text_vectorization_12/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2>
<sequential_24/text_vectorization_12/cond/strided_slice/stack?
>sequential_24/text_vectorization_12/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2@
>sequential_24/text_vectorization_12/cond/strided_slice/stack_1?
>sequential_24/text_vectorization_12/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_24/text_vectorization_12/cond/strided_slice/stack_2?
6sequential_24/text_vectorization_12/cond/strided_sliceStridedSlice~sequential_24_text_vectorization_12_cond_strided_slice_sequential_24_text_vectorization_12_raggedtotensor_raggedtensortotensorEsequential_24/text_vectorization_12/cond/strided_slice/stack:output:0Gsequential_24/text_vectorization_12/cond/strided_slice/stack_1:output:0Gsequential_24/text_vectorization_12/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask28
6sequential_24/text_vectorization_12/cond/strided_slice?
1sequential_24/text_vectorization_12/cond/IdentityIdentity?sequential_24/text_vectorization_12/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????23
1sequential_24/text_vectorization_12/cond/Identity"o
1sequential_24_text_vectorization_12_cond_identity:sequential_24/text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
.__inference_sequential_24_layer_call_fn_310362
text_vectorization_12_input
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_3103492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::22
StatefulPartitionedCallStatefulPartitionedCall:d `
'
_output_shapes
:?????????
5
_user_specified_nametext_vectorization_12_input:

_output_shapes
: 
?
e
F__inference_dropout_25_layer_call_and_return_conditional_losses_309878

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_25_layer_call_and_return_conditional_losses_310999

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_24_layer_call_and_return_conditional_losses_309841

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_25_layer_call_and_return_conditional_losses_309883

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
s
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_309795

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_310482
text_vectorization_12_input
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_3097792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::22
StatefulPartitionedCallStatefulPartitionedCall:d `
'
_output_shapes
:?????????
5
_user_specified_nametext_vectorization_12_input:

_output_shapes
: 
?

?
H__inference_embedding_12_layer_call_and_return_conditional_losses_309812

inputs
embedding_lookup_309806
identity??embedding_lookupf
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_309806Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/309806*4
_output_shapes"
 :??????????????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/309806*4
_output_shapes"
 :??????????????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
&text_vectorization_12_cond_true_310679G
Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub\
Xtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
+text_vectorization_12/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2-
+text_vectorization_12/cond/Pad/paddings/1/0?
)text_vectorization_12/cond/Pad/paddings/1Pack4text_vectorization_12/cond/Pad/paddings/1/0:output:0Ctext_vectorization_12_cond_pad_paddings_1_text_vectorization_12_sub*
N*
T0*
_output_shapes
:2+
)text_vectorization_12/cond/Pad/paddings/1?
+text_vectorization_12/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+text_vectorization_12/cond/Pad/paddings/0_1?
'text_vectorization_12/cond/Pad/paddingsPack4text_vectorization_12/cond/Pad/paddings/0_1:output:02text_vectorization_12/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2)
'text_vectorization_12/cond/Pad/paddings?
text_vectorization_12/cond/PadPadXtext_vectorization_12_cond_pad_text_vectorization_12_raggedtotensor_raggedtensortotensor0text_vectorization_12/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2 
text_vectorization_12/cond/Pad?
#text_vectorization_12/cond/IdentityIdentity'text_vectorization_12/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?:
?

__inference__traced_save_311176
file_prefix6
2savev2_embedding_12_embeddings_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableopV
Rsavev2_string_lookup_12_index_table_lookup_table_export_values_lookuptableexportv2X
Tsavev2_string_lookup_12_index_table_lookup_table_export_values_lookuptableexportv2_1	(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop=
9savev2_adam_embedding_12_embeddings_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop=
9savev2_adam_embedding_12_embeddings_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop
savev2_const_1

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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-valuesB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_embedding_12_embeddings_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableopRsavev2_string_lookup_12_index_table_lookup_table_export_values_lookuptableexportv2Tsavev2_string_lookup_12_index_table_lookup_table_export_values_lookuptableexportv2_1$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop9savev2_adam_embedding_12_embeddings_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop9savev2_adam_embedding_12_embeddings_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *'
dtypes
2		2
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
_input_shapesy
w: :	?N::::: : : : : : : : : : : : : :	?N:::	?N::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?N:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?N:$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	?N:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
d
+__inference_dropout_25_layer_call_fn_311009

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_3098782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
s
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_310982

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
__inference_restore_fn_311075
restored_tensors_0
restored_tensors_1	O
Kstring_lookup_12_index_table_table_restore_lookuptableimportv2_table_handle
identity??>string_lookup_12_index_table_table_restore/LookupTableImportV2?
>string_lookup_12_index_table_table_restore/LookupTableImportV2LookupTableImportV2Kstring_lookup_12_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2@
>string_lookup_12_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0?^string_lookup_12_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:::2?
>string_lookup_12_index_table_table_restore/LookupTableImportV2>string_lookup_12_index_table_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
d
F__inference_dropout_25_layer_call_and_return_conditional_losses_311004

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?g
?
"__inference__traced_restore_311255
file_prefix,
(assignvariableop_embedding_12_embeddings&
"assignvariableop_1_dense_12_kernel$
 assignvariableop_2_dense_12_bias_
[string_lookup_12_index_table_table_restore_lookuptableimportv2_string_lookup_12_index_table 
assignvariableop_3_adam_iter"
assignvariableop_4_adam_beta_1"
assignvariableop_5_adam_beta_2!
assignvariableop_6_adam_decay)
%assignvariableop_7_adam_learning_rate
assignvariableop_8_total
assignvariableop_9_count
assignvariableop_10_total_1
assignvariableop_11_count_1
assignvariableop_12_total_2
assignvariableop_13_count_2
assignvariableop_14_total_3
assignvariableop_15_count_36
2assignvariableop_16_adam_embedding_12_embeddings_m.
*assignvariableop_17_adam_dense_12_kernel_m,
(assignvariableop_18_adam_dense_12_bias_m6
2assignvariableop_19_adam_embedding_12_embeddings_v.
*assignvariableop_20_adam_dense_12_kernel_v,
(assignvariableop_21_adam_dense_12_bias_v
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?>string_lookup_12_index_table_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-valuesB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/0/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBatrainable_variables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp(assignvariableop_embedding_12_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_12_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_12_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2?
>string_lookup_12_index_table_table_restore/LookupTableImportV2LookupTableImportV2[string_lookup_12_index_table_table_restore_lookuptableimportv2_string_lookup_12_index_tableRestoreV2:tensors:3RestoreV2:tensors:4*	
Tin0*

Tout0	*/
_class%
#!loc:@string_lookup_12_index_table*
_output_shapes
 2@
>string_lookup_12_index_table_table_restore/LookupTableImportV2k

Identity_3IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7l

Identity_8IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8l

Identity_9IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_3Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_3Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp2assignvariableop_16_adam_embedding_12_embeddings_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_12_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_12_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_embedding_12_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_12_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_12_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp?^string_lookup_12_index_table_table_restore/LookupTableImportV2"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22?
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9?^string_lookup_12_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92?
>string_lookup_12_index_table_table_restore/LookupTableImportV2>string_lookup_12_index_table_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:51
/
_class%
#!loc:@string_lookup_12_index_table
?
?
'text_vectorization_12_cond_false_310229*
&text_vectorization_12_cond_placeholderf
btext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
.text_vectorization_12/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.text_vectorization_12/cond/strided_slice/stack?
0text_vectorization_12/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   22
0text_vectorization_12/cond/strided_slice/stack_1?
0text_vectorization_12/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0text_vectorization_12/cond/strided_slice/stack_2?
(text_vectorization_12/cond/strided_sliceStridedSlicebtext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor7text_vectorization_12/cond/strided_slice/stack:output:09text_vectorization_12/cond/strided_slice/stack_1:output:09text_vectorization_12/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2*
(text_vectorization_12/cond/strided_slice?
#text_vectorization_12/cond/IdentityIdentity1text_vectorization_12/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
J
.__inference_activation_11_layer_call_fn_310921

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_11_layer_call_and_return_conditional_losses_3101612
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
I__inference_sequential_24_layer_call_and_return_conditional_losses_310719

inputs`
\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handlea
]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value	6
2sequential_23_embedding_12_embedding_lookup_3107029
5sequential_23_dense_12_matmul_readvariableop_resource:
6sequential_23_dense_12_biasadd_readvariableop_resource
identity??-sequential_23/dense_12/BiasAdd/ReadVariableOp?,sequential_23/dense_12/MatMul/ReadVariableOp?+sequential_23/embedding_12/embedding_lookup?Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
!text_vectorization_12/StringLowerStringLowerinputs*'
_output_shapes
:?????????2#
!text_vectorization_12/StringLower?
(text_vectorization_12/StaticRegexReplaceStaticRegexReplace*text_vectorization_12/StringLower:output:0*'
_output_shapes
:?????????*
pattern<br />*
rewrite 2*
(text_vectorization_12/StaticRegexReplace?
*text_vectorization_12/StaticRegexReplace_1StaticRegexReplace1text_vectorization_12/StaticRegexReplace:output:0*'
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite 2,
*text_vectorization_12/StaticRegexReplace_1?
text_vectorization_12/SqueezeSqueeze3text_vectorization_12/StaticRegexReplace_1:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
text_vectorization_12/Squeeze?
'text_vectorization_12/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2)
'text_vectorization_12/StringSplit/Const?
/text_vectorization_12/StringSplit/StringSplitV2StringSplitV2&text_vectorization_12/Squeeze:output:00text_vectorization_12/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:21
/text_vectorization_12/StringSplit/StringSplitV2?
5text_vectorization_12/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5text_vectorization_12/StringSplit/strided_slice/stack?
7text_vectorization_12/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7text_vectorization_12/StringSplit/strided_slice/stack_1?
7text_vectorization_12/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7text_vectorization_12/StringSplit/strided_slice/stack_2?
/text_vectorization_12/StringSplit/strided_sliceStridedSlice9text_vectorization_12/StringSplit/StringSplitV2:indices:0>text_vectorization_12/StringSplit/strided_slice/stack:output:0@text_vectorization_12/StringSplit/strided_slice/stack_1:output:0@text_vectorization_12/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask21
/text_vectorization_12/StringSplit/strided_slice?
7text_vectorization_12/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7text_vectorization_12/StringSplit/strided_slice_1/stack?
9text_vectorization_12/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_1?
9text_vectorization_12/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9text_vectorization_12/StringSplit/strided_slice_1/stack_2?
1text_vectorization_12/StringSplit/strided_slice_1StridedSlice7text_vectorization_12/StringSplit/StringSplitV2:shape:0@text_vectorization_12/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_12/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask23
1text_vectorization_12/StringSplit/strided_slice_1?
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_12/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Z
Xtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_12/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2h
ftext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2c
atext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2d
btext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2b
`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2f
dtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount\text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2g
etext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2e
ctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2a
_text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2\
Ztext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2LookupTableFindV2\text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_table_handle8text_vectorization_12/StringSplit/StringSplitV2:values:0]text_vectorization_12_string_lookup_12_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2Q
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2?
8text_vectorization_12/string_lookup_12/assert_equal/NoOpNoOp*
_output_shapes
 2:
8text_vectorization_12/string_lookup_12/assert_equal/NoOp?
/text_vectorization_12/string_lookup_12/IdentityIdentityXtext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????21
/text_vectorization_12/string_lookup_12/Identity?
1text_vectorization_12/string_lookup_12/Identity_1Identityctext_vectorization_12/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????23
1text_vectorization_12/string_lookup_12/Identity_1?
2text_vectorization_12/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 24
2text_vectorization_12/RaggedToTensor/default_value?
*text_vectorization_12/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2,
*text_vectorization_12/RaggedToTensor/Const?
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_12/RaggedToTensor/Const:output:08text_vectorization_12/string_lookup_12/Identity:output:0;text_vectorization_12/RaggedToTensor/default_value:output:0:text_vectorization_12/string_lookup_12/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2;
9text_vectorization_12/RaggedToTensor/RaggedTensorToTensor?
text_vectorization_12/ShapeShapeBtext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
text_vectorization_12/Shape?
)text_vectorization_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)text_vectorization_12/strided_slice/stack?
+text_vectorization_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_1?
+text_vectorization_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+text_vectorization_12/strided_slice/stack_2?
#text_vectorization_12/strided_sliceStridedSlice$text_vectorization_12/Shape:output:02text_vectorization_12/strided_slice/stack:output:04text_vectorization_12/strided_slice/stack_1:output:04text_vectorization_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#text_vectorization_12/strided_slice}
text_vectorization_12/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/sub/x?
text_vectorization_12/subSub$text_vectorization_12/sub/x:output:0,text_vectorization_12/strided_slice:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/sub
text_vectorization_12/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
text_vectorization_12/Less/y?
text_vectorization_12/LessLess,text_vectorization_12/strided_slice:output:0%text_vectorization_12/Less/y:output:0*
T0*
_output_shapes
: 2
text_vectorization_12/Less?
text_vectorization_12/condStatelessIftext_vectorization_12/Less:z:0text_vectorization_12/sub:z:0Btext_vectorization_12/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'text_vectorization_12_cond_false_310680*/
output_shapes
:??????????????????*9
then_branch*R(
&text_vectorization_12_cond_true_3106792
text_vectorization_12/cond?
#text_vectorization_12/cond/IdentityIdentity#text_vectorization_12/cond:output:0*
T0	*(
_output_shapes
:??????????2%
#text_vectorization_12/cond/Identity?
sequential_23/CastCast,text_vectorization_12/cond/Identity:output:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????2
sequential_23/Cast?
sequential_23/embedding_12/CastCastsequential_23/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:??????????2!
sequential_23/embedding_12/Cast?
+sequential_23/embedding_12/embedding_lookupResourceGather2sequential_23_embedding_12_embedding_lookup_310702#sequential_23/embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*E
_class;
97loc:@sequential_23/embedding_12/embedding_lookup/310702*,
_output_shapes
:??????????*
dtype02-
+sequential_23/embedding_12/embedding_lookup?
4sequential_23/embedding_12/embedding_lookup/IdentityIdentity4sequential_23/embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@sequential_23/embedding_12/embedding_lookup/310702*,
_output_shapes
:??????????26
4sequential_23/embedding_12/embedding_lookup/Identity?
6sequential_23/embedding_12/embedding_lookup/Identity_1Identity=sequential_23/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????28
6sequential_23/embedding_12/embedding_lookup/Identity_1?
!sequential_23/dropout_24/IdentityIdentity?sequential_23/embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2#
!sequential_23/dropout_24/Identity?
@sequential_23/global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2B
@sequential_23/global_average_pooling1d_12/Mean/reduction_indices?
.sequential_23/global_average_pooling1d_12/MeanMean*sequential_23/dropout_24/Identity:output:0Isequential_23/global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????20
.sequential_23/global_average_pooling1d_12/Mean?
!sequential_23/dropout_25/IdentityIdentity7sequential_23/global_average_pooling1d_12/Mean:output:0*
T0*'
_output_shapes
:?????????2#
!sequential_23/dropout_25/Identity?
,sequential_23/dense_12/MatMul/ReadVariableOpReadVariableOp5sequential_23_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_23/dense_12/MatMul/ReadVariableOp?
sequential_23/dense_12/MatMulMatMul*sequential_23/dropout_25/Identity:output:04sequential_23/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_23/dense_12/MatMul?
-sequential_23/dense_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_23_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_23/dense_12/BiasAdd/ReadVariableOp?
sequential_23/dense_12/BiasAddBiasAdd'sequential_23/dense_12/MatMul:product:05sequential_23/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_23/dense_12/BiasAdd?
activation_11/SigmoidSigmoid'sequential_23/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
activation_11/Sigmoid?
IdentityIdentityactivation_11/Sigmoid:y:0.^sequential_23/dense_12/BiasAdd/ReadVariableOp-^sequential_23/dense_12/MatMul/ReadVariableOp,^sequential_23/embedding_12/embedding_lookupP^text_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????:: :::2^
-sequential_23/dense_12/BiasAdd/ReadVariableOp-sequential_23/dense_12/BiasAdd/ReadVariableOp2\
,sequential_23/dense_12/MatMul/ReadVariableOp,sequential_23/dense_12/MatMul/ReadVariableOp2Z
+sequential_23/embedding_12/embedding_lookup+sequential_23/embedding_12/embedding_lookup2?
Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2Otext_vectorization_12/string_lookup_12/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
'text_vectorization_12_cond_false_310680*
&text_vectorization_12_cond_placeholderf
btext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
.text_vectorization_12/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.text_vectorization_12/cond/strided_slice/stack?
0text_vectorization_12/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   22
0text_vectorization_12/cond/strided_slice/stack_1?
0text_vectorization_12/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0text_vectorization_12/cond/strided_slice/stack_2?
(text_vectorization_12/cond/strided_sliceStridedSlicebtext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor7text_vectorization_12/cond/strided_slice/stack:output:09text_vectorization_12/cond/strided_slice/stack_1:output:09text_vectorization_12/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2*
(text_vectorization_12/cond/strided_slice?
#text_vectorization_12/cond/IdentityIdentity1text_vectorization_12/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?

?
H__inference_embedding_12_layer_call_and_return_conditional_losses_310931

inputs
embedding_lookup_310925
identity??embedding_lookupf
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_310925Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/310925*4
_output_shapes"
 :??????????????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/310925*4
_output_shapes"
 :??????????????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_23_layer_call_and_return_conditional_losses_309938
embedding_12_input
embedding_12_309926
dense_12_309932
dense_12_309934
identity?? dense_12/StatefulPartitionedCall?$embedding_12/StatefulPartitionedCall?
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallembedding_12_inputembedding_12_309926*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_3098122&
$embedding_12/StatefulPartitionedCall?
dropout_24/PartitionedCallPartitionedCall-embedding_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_24_layer_call_and_return_conditional_losses_3098412
dropout_24/PartitionedCall?
+global_average_pooling1d_12/PartitionedCallPartitionedCall#dropout_24/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_3098592-
+global_average_pooling1d_12/PartitionedCall?
dropout_25/PartitionedCallPartitionedCall4global_average_pooling1d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_3098832
dropout_25/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dense_12_309932dense_12_309934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3099062"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall:d `
0
_output_shapes
:??????????????????
,
_user_specified_nameembedding_12_input
?
e
F__inference_dropout_24_layer_call_and_return_conditional_losses_309836

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
M
__inference__creator_311038
identity??string_lookup_12_index_table?
string_lookup_12_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_286737*
value_dtype0	2
string_lookup_12_index_table?
IdentityIdentity+string_lookup_12_index_table:table_handle:0^string_lookup_12_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 2<
string_lookup_12_index_tablestring_lookup_12_index_table
?
s
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_310971

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
'text_vectorization_12_cond_false_310320*
&text_vectorization_12_cond_placeholderf
btext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor	'
#text_vectorization_12_cond_identity	?
.text_vectorization_12/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.text_vectorization_12/cond/strided_slice/stack?
0text_vectorization_12/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   22
0text_vectorization_12/cond/strided_slice/stack_1?
0text_vectorization_12/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0text_vectorization_12/cond/strided_slice/stack_2?
(text_vectorization_12/cond/strided_sliceStridedSlicebtext_vectorization_12_cond_strided_slice_text_vectorization_12_raggedtotensor_raggedtensortotensor7text_vectorization_12/cond/strided_slice/stack:output:09text_vectorization_12/cond/strided_slice/stack_1:output:09text_vectorization_12/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2*
(text_vectorization_12/cond/strided_slice?
#text_vectorization_12/cond/IdentityIdentity1text_vectorization_12/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2%
#text_vectorization_12/cond/Identity"S
#text_vectorization_12_cond_identity,text_vectorization_12/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
__inference_save_fn_311067
checkpoint_key\
Xstring_lookup_12_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Kstring_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2?
Kstring_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Xstring_lookup_12_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2M
Kstring_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0L^string_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0L^string_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityRstring_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2:keys:0L^string_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
:2

Identity_2?

Identity_3Identity	add_1:z:0L^string_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0L^string_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityTstring_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2:values:0L^string_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Kstring_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2Kstring_lookup_12_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
I__inference_sequential_23_layer_call_and_return_conditional_losses_310126

inputs	(
$embedding_12_embedding_lookup_310110+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?embedding_12/embedding_lookup^
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????2
Castz
embedding_12/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:??????????2
embedding_12/Cast?
embedding_12/embedding_lookupResourceGather$embedding_12_embedding_lookup_310110embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_12/embedding_lookup/310110*,
_output_shapes
:??????????*
dtype02
embedding_12/embedding_lookup?
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_12/embedding_lookup/310110*,
_output_shapes
:??????????2(
&embedding_12/embedding_lookup/Identity?
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_12/embedding_lookup/Identity_1?
dropout_24/IdentityIdentity1embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
dropout_24/Identity?
2global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2global_average_pooling1d_12/Mean/reduction_indices?
 global_average_pooling1d_12/MeanMeandropout_24/Identity:output:0;global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2"
 global_average_pooling1d_12/Mean?
dropout_25/IdentityIdentity)global_average_pooling1d_12/Mean:output:0*
T0*'
_output_shapes
:?????????2
dropout_25/Identity?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_25/Identity:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd?
IdentityIdentitydense_12/BiasAdd:output:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp^embedding_12/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????:::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2>
embedding_12/embedding_lookupembedding_12/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_23_layer_call_fn_310900

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_3099562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_23_layer_call_and_return_conditional_losses_309923
embedding_12_input
embedding_12_309821
dense_12_309917
dense_12_309919
identity?? dense_12/StatefulPartitionedCall?"dropout_24/StatefulPartitionedCall?"dropout_25/StatefulPartitionedCall?$embedding_12/StatefulPartitionedCall?
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallembedding_12_inputembedding_12_309821*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_3098122&
$embedding_12/StatefulPartitionedCall?
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall-embedding_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_24_layer_call_and_return_conditional_losses_3098362$
"dropout_24/StatefulPartitionedCall?
+global_average_pooling1d_12/PartitionedCallPartitionedCall+dropout_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_3098592-
+global_average_pooling1d_12/PartitionedCall?
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_12/PartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_3098782$
"dropout_25/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dense_12_309917dense_12_309919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3099062"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall:d `
0
_output_shapes
:??????????????????
,
_user_specified_nameembedding_12_input
?
d
+__inference_dropout_24_layer_call_fn_310960

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_24_layer_call_and_return_conditional_losses_3098362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_23_layer_call_fn_310911

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_3099822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_23_layer_call_and_return_conditional_losses_310889

inputs(
$embedding_12_embedding_lookup_310873+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?embedding_12/embedding_lookup?
embedding_12/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:??????????????????2
embedding_12/Cast?
embedding_12/embedding_lookupResourceGather$embedding_12_embedding_lookup_310873embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_12/embedding_lookup/310873*4
_output_shapes"
 :??????????????????*
dtype02
embedding_12/embedding_lookup?
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_12/embedding_lookup/310873*4
_output_shapes"
 :??????????????????2(
&embedding_12/embedding_lookup/Identity?
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????2*
(embedding_12/embedding_lookup/Identity_1?
dropout_24/IdentityIdentity1embedding_12/embedding_lookup/Identity_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout_24/Identity?
2global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2global_average_pooling1d_12/Mean/reduction_indices?
 global_average_pooling1d_12/MeanMeandropout_24/Identity:output:0;global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2"
 global_average_pooling1d_12/Mean?
dropout_25/IdentityIdentity)global_average_pooling1d_12/Mean:output:0*
T0*'
_output_shapes
:?????????2
dropout_25/Identity?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_25/Identity:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd?
IdentityIdentitydense_12/BiasAdd:output:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp^embedding_12/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2>
embedding_12/embedding_lookupembedding_12/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
/
__inference__initializer_311043
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
c
text_vectorization_12_inputD
-serving_default_text_vectorization_12_input:0?????????A
activation_110
StatefulPartitionedCall:0?????????tensorflow/serving/predict:â
?-
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
~__call__
_default_save_signature
+?&call_and_return_all_conditional_losses"?+
_tf_keras_sequential?+{"class_name": "Sequential", "name": "sequential_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "text_vectorization_12_input"}}, {"class_name": "TextVectorization", "config": {"name": "text_vectorization_12", "trainable": true, "dtype": "string", "max_tokens": 10000, "standardize": "customStandardization", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 250, "pad_to_max_tokens": true}}, {"class_name": "Sequential", "config": {"name": "sequential_23", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_12_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10001, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}, {"class_name": "Dropout", "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "text_vectorization_12_input"}}, {"class_name": "TextVectorization", "config": {"name": "text_vectorization_12", "trainable": true, "dtype": "string", "max_tokens": 10000, "standardize": "customStandardization", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 250, "pad_to_max_tokens": true}}, {"class_name": "Sequential", "config": {"name": "sequential_23", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_12_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10001, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}, {"class_name": "Dropout", "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?

state_variables
_index_lookup_layer
	keras_api"?
_tf_keras_layer?{"class_name": "TextVectorization", "name": "text_vectorization_12", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "text_vectorization_12", "trainable": true, "dtype": "string", "max_tokens": 10000, "standardize": "customStandardization", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 250, "pad_to_max_tokens": true}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?"
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"? 
_tf_keras_sequential? {"class_name": "Sequential", "name": "sequential_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_23", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_12_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10001, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}, {"class_name": "Dropout", "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_23", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_12_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10001, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}, {"class_name": "Dropout", "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": true, "label_smoothing": 0}}, "metrics": [[{"class_name": "BinaryAccuracy", "config": {"name": "binary_accuracy", "dtype": "float32", "threshold": 0.0}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}
"
	optimizer
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
1
2
3"
trackable_list_wrapper
?
layer_regularization_losses
trainable_variables
layer_metrics
 non_trainable_variables

!layers
regularization_losses
"metrics
	variables
~__call__
_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
?
#state_variables

$_table
%	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup_12", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "string", "invert": false, "max_tokens": 10000, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
"
_generic_user_object
?

embeddings
&trainable_variables
'regularization_losses
(	variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10001, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}}
?
*trainable_variables
+regularization_losses
,	variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
.trainable_variables
/regularization_losses
0	variables
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
2trainable_variables
3regularization_losses
4	variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

kernel
bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
:iter

;beta_1

<beta_2
	=decay
>learning_ratemxmymzv{v|v}"
	optimizer
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
?layer_regularization_losses
trainable_variables
@layer_metrics
Anon_trainable_variables

Blayers
regularization_losses
Cmetrics
	variables
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
Dlayer_regularization_losses
trainable_variables
Elayer_metrics
Fnon_trainable_variables

Glayers
regularization_losses
Hmetrics
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(	?N2embedding_12/embeddings
!:2dense_12/kernel
:2dense_12/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_dict_wrapper
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
Klayer_regularization_losses
&trainable_variables
Llayer_metrics
Mnon_trainable_variables

Nlayers
'regularization_losses
Ometrics
(	variables
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
Player_regularization_losses
*trainable_variables
Qlayer_metrics
Rnon_trainable_variables

Slayers
+regularization_losses
Tmetrics
,	variables
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
Ulayer_regularization_losses
.trainable_variables
Vlayer_metrics
Wnon_trainable_variables

Xlayers
/regularization_losses
Ymetrics
0	variables
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
Zlayer_regularization_losses
2trainable_variables
[layer_metrics
\non_trainable_variables

]layers
3regularization_losses
^metrics
4	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
_layer_regularization_losses
6trainable_variables
`layer_metrics
anon_trainable_variables

blayers
7regularization_losses
cmetrics
8	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
d0
e1"
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
?
	ftotal
	gcount
h	variables
i	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
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
?
	ototal
	pcount
q	variables
r	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	stotal
	tcount
u
_fn_kwargs
v	variables
w	keras_api"?
_tf_keras_metric?{"class_name": "BinaryAccuracy", "name": "binary_accuracy", "dtype": "float32", "config": {"name": "binary_accuracy", "dtype": "float32", "threshold": 0.0}}
:  (2total
:  (2count
.
f0
g1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
j0
k1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
:  (2total
:  (2count
.
o0
p1"
trackable_list_wrapper
-
q	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
s0
t1"
trackable_list_wrapper
-
v	variables"
_generic_user_object
/:-	?N2Adam/embedding_12/embeddings/m
&:$2Adam/dense_12/kernel/m
 :2Adam/dense_12/bias/m
/:-	?N2Adam/embedding_12/embeddings/v
&:$2Adam/dense_12/kernel/v
 :2Adam/dense_12/bias/v
?2?
.__inference_sequential_24_layer_call_fn_310362
.__inference_sequential_24_layer_call_fn_310465
.__inference_sequential_24_layer_call_fn_310749
.__inference_sequential_24_layer_call_fn_310734?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_309779?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *:?7
5?2
text_vectorization_12_input?????????
?2?
I__inference_sequential_24_layer_call_and_return_conditional_losses_310621
I__inference_sequential_24_layer_call_and_return_conditional_losses_310719
I__inference_sequential_24_layer_call_and_return_conditional_losses_310258
I__inference_sequential_24_layer_call_and_return_conditional_losses_310170?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference_save_fn_311067checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_311075restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
?2?
.__inference_sequential_23_layer_call_fn_310835
.__inference_sequential_23_layer_call_fn_309991
.__inference_sequential_23_layer_call_fn_309965
.__inference_sequential_23_layer_call_fn_310824
.__inference_sequential_23_layer_call_fn_310911
.__inference_sequential_23_layer_call_fn_310900?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_23_layer_call_and_return_conditional_losses_310792
I__inference_sequential_23_layer_call_and_return_conditional_losses_310813
I__inference_sequential_23_layer_call_and_return_conditional_losses_310869
I__inference_sequential_23_layer_call_and_return_conditional_losses_310889
I__inference_sequential_23_layer_call_and_return_conditional_losses_309923
I__inference_sequential_23_layer_call_and_return_conditional_losses_309938?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_activation_11_layer_call_fn_310921?
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
?2?
I__inference_activation_11_layer_call_and_return_conditional_losses_310916?
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
?B?
$__inference_signature_wrapper_310482text_vectorization_12_input"?
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
?2?
-__inference_embedding_12_layer_call_fn_310938?
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
?2?
H__inference_embedding_12_layer_call_and_return_conditional_losses_310931?
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
?2?
+__inference_dropout_24_layer_call_fn_310960
+__inference_dropout_24_layer_call_fn_310965?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_24_layer_call_and_return_conditional_losses_310955
F__inference_dropout_24_layer_call_and_return_conditional_losses_310950?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
<__inference_global_average_pooling1d_12_layer_call_fn_310976
<__inference_global_average_pooling1d_12_layer_call_fn_310987?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_310982
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_310971?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dropout_25_layer_call_fn_311014
+__inference_dropout_25_layer_call_fn_311009?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_25_layer_call_and_return_conditional_losses_310999
F__inference_dropout_25_layer_call_and_return_conditional_losses_311004?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_12_layer_call_fn_311033?
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
?2?
D__inference_dense_12_layer_call_and_return_conditional_losses_311024?
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
?2?
__inference__creator_311038?
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
annotations? *? 
?2?
__inference__initializer_311043?
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
annotations? *? 
?2?
__inference__destroyer_311048?
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
annotations? *? 
	J
Const7
__inference__creator_311038?

? 
? "? 9
__inference__destroyer_311048?

? 
? "? ;
__inference__initializer_311043?

? 
? "? ?
!__inference__wrapped_model_309779?$?D?A
:?7
5?2
text_vectorization_12_input?????????
? "=?:
8
activation_11'?$
activation_11??????????
I__inference_activation_11_layer_call_and_return_conditional_losses_310916X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
.__inference_activation_11_layer_call_fn_310921K/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_dense_12_layer_call_and_return_conditional_losses_311024\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_12_layer_call_fn_311033O/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_dropout_24_layer_call_and_return_conditional_losses_310950v@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
F__inference_dropout_24_layer_call_and_return_conditional_losses_310955v@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
+__inference_dropout_24_layer_call_fn_310960i@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
+__inference_dropout_24_layer_call_fn_310965i@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
F__inference_dropout_25_layer_call_and_return_conditional_losses_310999\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
F__inference_dropout_25_layer_call_and_return_conditional_losses_311004\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ~
+__inference_dropout_25_layer_call_fn_311009O3?0
)?&
 ?
inputs?????????
p
? "??????????~
+__inference_dropout_25_layer_call_fn_311014O3?0
)?&
 ?
inputs?????????
p 
? "???????????
H__inference_embedding_12_layer_call_and_return_conditional_losses_310931q8?5
.?+
)?&
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
-__inference_embedding_12_layer_call_fn_310938d8?5
.?+
)?&
inputs??????????????????
? "%?"???????????????????
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_310971i@?=
6?3
-?*
inputs??????????????????

 
? "%?"
?
0?????????
? ?
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_310982{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
<__inference_global_average_pooling1d_12_layer_call_fn_310976\@?=
6?3
-?*
inputs??????????????????

 
? "???????????
<__inference_global_average_pooling1d_12_layer_call_fn_310987nI?F
??<
6?3
inputs'???????????????????????????

 
? "!???????????????????z
__inference_restore_fn_311075Y$K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_311067?$&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
I__inference_sequential_23_layer_call_and_return_conditional_losses_309923zL?I
B??
5?2
embedding_12_input??????????????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_23_layer_call_and_return_conditional_losses_309938zL?I
B??
5?2
embedding_12_input??????????????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_23_layer_call_and_return_conditional_losses_310792f8?5
.?+
!?
inputs??????????	
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_23_layer_call_and_return_conditional_losses_310813f8?5
.?+
!?
inputs??????????	
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_23_layer_call_and_return_conditional_losses_310869n@?=
6?3
)?&
inputs??????????????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_23_layer_call_and_return_conditional_losses_310889n@?=
6?3
)?&
inputs??????????????????
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_23_layer_call_fn_309965mL?I
B??
5?2
embedding_12_input??????????????????
p

 
? "???????????
.__inference_sequential_23_layer_call_fn_309991mL?I
B??
5?2
embedding_12_input??????????????????
p 

 
? "???????????
.__inference_sequential_23_layer_call_fn_310824Y8?5
.?+
!?
inputs??????????	
p

 
? "???????????
.__inference_sequential_23_layer_call_fn_310835Y8?5
.?+
!?
inputs??????????	
p 

 
? "???????????
.__inference_sequential_23_layer_call_fn_310900a@?=
6?3
)?&
inputs??????????????????
p

 
? "???????????
.__inference_sequential_23_layer_call_fn_310911a@?=
6?3
)?&
inputs??????????????????
p 

 
? "???????????
I__inference_sequential_24_layer_call_and_return_conditional_losses_310170}$?L?I
B??
5?2
text_vectorization_12_input?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_24_layer_call_and_return_conditional_losses_310258}$?L?I
B??
5?2
text_vectorization_12_input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_24_layer_call_and_return_conditional_losses_310621h$?7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_24_layer_call_and_return_conditional_losses_310719h$?7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_24_layer_call_fn_310362p$?L?I
B??
5?2
text_vectorization_12_input?????????
p

 
? "???????????
.__inference_sequential_24_layer_call_fn_310465p$?L?I
B??
5?2
text_vectorization_12_input?????????
p 

 
? "???????????
.__inference_sequential_24_layer_call_fn_310734[$?7?4
-?*
 ?
inputs?????????
p

 
? "???????????
.__inference_sequential_24_layer_call_fn_310749[$?7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
$__inference_signature_wrapper_310482?$?c?`
? 
Y?V
T
text_vectorization_12_input5?2
text_vectorization_12_input?????????"=?:
8
activation_11'?$
activation_11?????????
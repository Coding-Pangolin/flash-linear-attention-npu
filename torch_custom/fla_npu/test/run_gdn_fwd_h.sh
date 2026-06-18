batch=1 #309
seqlen=16384
kNumHead=16
vNumHead=32
kHeadDim=128
vHeadDim=128
isVariedLen=1 #1
chunkSize=64
dtype="bf16"
useActualInput=0
useActualOutput=0
device=${TEST_DEVICE_ID:-2}
gDType="float"
stateDType="float"

useInitialState=0
storeFinalState=0
dataPath="/path/to/data"

echo 'Case: batch=' $batch ' seqlen=' $seqlen ' kNumHead=' $kNumHead  ' vNumHead=' $vNumHead ' kHeadDim=' $kHeadDim ' vHeadDim=' $vHeadDim ' isVariedLen=' $isVariedLen ' chunkSize=' $chunkSize ' dtype=' $dtype
python3 test_fwd_h.py $batch $seqlen $kNumHead $vNumHead $kHeadDim $vHeadDim $isVariedLen $chunkSize $useInitialState $storeFinalState "$dtype" $useActualInput $useActualOutput $dataPath $device $gDType $stateDType
# python data_compare_h.py $dtype
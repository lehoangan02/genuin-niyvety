# Instruction

## DATA folder structure
```
root/
    DATA/
        label.txt
        frames/*.jpg
        iamges/*.jpg
        embeddings_SPA/*.npy
        embeddings_mobile-clip-CLT/*.npy
        embeddings_clip-vit-base-patch32/*.py
```

## How to set up the environment
```
pip install -r requirement_clip.txt
```

## How to train the model
```
# use requirement_clip.txt to set up the environment
cd AnTools
python main.py --phase train
```

## How to encode images
```
# for mobile-clip and clip-vit32, use requirement_clip.txt to set up the environment
cd AnTools
python main.py --phase encode --encoder clip-vit-base-patch32
# for SPA, use requirement_spa.txt to set up the environment
cd AnTools
python main.py --phase encode --encoder SPA
```

## How to test models
```
# set up the environment
cd AnTools
# write your own sanity check code and put them in the sanity_check folder for better orginization
python ./sanity_check/sanity_check_v3.py
```

## How to evaluate models
```
# set up the environment
cd AnTools
python main.py --phase test --resume ./weights/model_last.pth --batch_size 1 --num_workers 4
```
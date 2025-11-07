> [!WARNING] 
> Make sure your current work directory is at here, ITRI-GraspGen/models/ 
#### GraspGenModels
```bash
git clone https://huggingface.co/adithyamurali/GraspGenModels
```

#### SAM2Models
```bash
mkdir SAM2Models
cp ../Third_Party/sam2/checkpoints/download_ckpts.sh ./SAM2Models
cd SAM2Models && ./download_ckpts.sh && cd ..
```

#### FoundationStereoModels
```bash
uv run gdown --folder https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf
mv pretrained_models FoundationStereoModels
```

#### GroundingDINOModels
```bash
mkdir GroundingDINOModels
cd GroundingDINOModels && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth && cd ..
```

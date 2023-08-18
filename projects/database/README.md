# Dataset 

### `train/val` dataset (2,500 time points)
    squential dataset depending on the followings :
        [yaw damper (30, 40, 50, 70, 100)]
        ["curved", "straight"]
### `test` dataset (500 time points)
    squential dataset의 마지막 time points

## Overall Dataset
<img src="images/dataset.png">

## Train/Val dataset `fold split` strategy
- 학습 데이터셋 중간에 검증 데이터셋을 설정하면 연속성이 끊어지기 때문에 train/val dataset의 *앞쪽* 또는 *뒷쪽*을 validation dataset으로 정의
- train/val dataset의 `y` plot을 고려하면 class imbalance 문제가 다소 내포되어 있는 문제로 여겨짐.
- Leave-one-out을 활용하되 아래 그림과 같이 정의하고 변수 `N`, `H`를 통해서 fold split 및 모델의 입출력을 정의
    - `val-pre-N-H`
    - `val-post-N-H`
- 최종적으로 획득되는 모델들의 앙상블 모델을 제안해보면 좋을 것으로 사료됨.

<img src="images/fold_pre.png">
<img src="images/fold_post.png">

## Input & Output

<img src="images/input_output.png">
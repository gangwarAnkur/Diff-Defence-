# Diff-Defence

Test Instructions
Make sure to run the setup file to create all the necessary folders!

python setup.py 

Train the classifiers!
First of all, you must train the classifiers to test diffDefence. The training can be logged with wandb, set --wandb='run' if you want to log.

#python train.py --dataset='MNIST' --classifier='classifier_a' --model_title='c_a'
python train.py --dataset='MNIST' --classifier='classifier_b' --model_title='c_b'

Tests presented in the paper use classifier_a as the main classifier and classifier_b as the substitute model to create adversarial images to attack classifier_a in the black box settings. You must train both the classifier, and afterwards, you can choose what to use as the substitute model.

To train adversarial trained classifier, set --adv_train.

python train.py --dataset='MNIST' --adv_train=True --classifier='classifier_a' --model_title='c_a_adv'

Check the train.py file to set other configuration parameters.

Train the Diffusion Model
Afterwards, train the Diffusion model. Check the DDPM.py file to set other configuration parameters.

python DDPM.py --dataset='MNIST' --model_title='diff'

Samples images are saved on the imgs folder

Test DiffDefence
Finally, test DiffDefence!

White box attack setting

python diffDefence.py --dataset='MNIST' --classifier='c_a' --typeA='classifier_a' --diffusion_name='diff' --test_size=1000

Black box attack setting

python diffDefence.py --dataset='MNIST' --classifier='c_a' --typeA='classifier_a' --sub_classifier='c_b' --typeB='classifier_b'  --diffusion_name='diff' --test_size=1000

If you want to test adversarial trained model against adversarial attacks
$python diffDefence.py --dataset='MNIST' --classifier='c_a_adv' --typeA='classifier_a' --diffusion_name='diff' --test_size=1000

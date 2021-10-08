

if [ $# -eq 0 ]
then
	git push origin main:main
	git push git@github.com:zhaoedf/demo_pytorch_lightning.git main:demo_segmentation
else
	git push origin $1:$1
fi

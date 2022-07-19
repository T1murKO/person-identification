# Deploy to Torch Serve

## To start pytorch inference server run:

```
docker build -t ubuntu-torchserve:latest . 
run --rm --name torchserve_docker \  
           -p8080:8080 -p8081:8081 -p8082:8082 \  
           ubuntu-torchserve:latest \  
           torchserve --model-store /home/model-server/model-store/
		   --models net=net.mar
```



## To create new .mar model file.
1) Go to model_export folder.
2) Place model.pth file inside the folder
3) Run export.py
4) Run command
`torch-model-archiver --model-name net --version 1.0 --serialized-file net.pt --handler handler.py`
5) Place file net.mar to model store
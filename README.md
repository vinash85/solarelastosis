# PROJECT-Solar Elastosis

## CLAM has been cloned from [here] (https://github.com/mahmoodlab/CLAM)
1. All license and copy rights to the Original CLAM belongs to the [original authors] (https://www.nature.com/articles/s41551-020-00682-w) from Dr.Faisal Mahmood Lab [http://clam.mahmoodlab.org/]
2. Usage instruction for CLAM remains same as the instructions above
3. The additional code useage for multimodal data is given below

## Requirements
Docker usage:
1. cd solarelatosis/docker
2. Create image-command: "docker build . -t solarel:v1 --rm"
3. Run the image[port 9090 for jupyter notebooks]-command: "docker run --gpus=all --restart=always --shm-size=10g --name solarelastosis -p 9090:8888 -v path_on_local_system/:/home/tailab/solarelastosis -v path_on_local_system/:/home/tailab/data/  -dit solarel:v1"
4. Enter the docker:

Container to be released on docker hub soon

## Folder Structure in docker
/home/tailab

           |

            solarelastosis -- code

           |               |__docker
            
            Data -- gem_blgm
                   
                   |__gem_results      

fileId=1Uef-pOVOEHGVE8SpEBumACiK_RwUbb04 
fileName=final_model.pth		 
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null  
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  					
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

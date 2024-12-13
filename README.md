### Train and save the model 
1. Train the model and copy the model.pkl to the deployment directory.
2. Use docker to deploy the model as a web app service
- Build your docker image and run the model locally

+ docker build -t healthmodel.azurecr.io/health-insurance:latest .

+ docker run -d -p 5000:5000 healthmodel.azurecr.io/health-insurance:latest 

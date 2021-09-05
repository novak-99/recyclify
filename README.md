# Recyclify

## Inspiration
Only 9% of plastic alone has been recycled. This is significant as the evasion of proper recycling habits causes habitat destruction, pollution, and the emission of greenhouse gases which contributes to climate change. This is let alone the threat posed to the sustainability of resources, as finite materials which could be recycled and reused are needlessly wasted as they accumulate in landfills. Interestingly enough, it has been found that the reasons for low rates of recycling include apathy, carelessness, and ambiguity. With the aid of software development and AI, we knew something could be done to prevent low rates of recycling. 

## What it does
Our program consists of deployments onto both a website as well as mobile applications for Android and iOS. 

Our mobile application allows users to take and upload photos of images of waste products and determine what material they are and whether they are recyclable or not. This solves the problem of ambiguity and confusion surrounding recycling as the results of the particular scan will indicate whether or not the material is recyclable. Furthermore, our includes a points system in which users can add points to their profile when successfully scanning recyclable items. Our app also contains a leaderboard of other users ranked by the number of points they have to encourage friendly competition and engagement surrounding recycling. This solves the problem of apathy as for those not already interested in the cause, the gamification of the idea of recycling adds an extra incentive.

Unfortunately, due to time constraints, we were not able to integrate our AI model with our mobile app and it currently uses a simulation for predictions. However, our models are fully functional and we have made deployments to the macOS and Windows operating systems. We plan to continue improving upon mobile deployments in the future. 

Our website includes general information about the basics of recycling, general information about Recyclify, and its founders. It also contains download links for both Windows and MacOS deployments of an auxiliary computer application in the Recyclify suite. This application, in particular, allows governments to determine the percentage of waste treated by recycling by country based on widely available data pertaining to waste. The resulting predictions are highly accurate and precise, and this allows governments and political actors to enact environmentally conscious legislation appropriately. 

## How we built it
The website was built with JavaScript, HTML5, CSS3, and hosted on the Qoom framework. Our mobile application was built using Flutter and Dart. 

Our models used for visually detecting the type of trash bin and recycling material were built using PyTorch. As for the algorithms themselves, we utilized the publicly available ResNet-18 visual Convolutional Neural Network (CNN) model and performed training using the Adam optimizer. We utilized Google Colab's access to Graphics Processing Units (GPUs) to accelerate training time and model development. 

Our model for detecting the percentage of waste treated by recycling was built using the Sci-Kit Learn framework. We performed Exploratory Data Analysis (EDA) on a publicly available dataset in which we removed extraneous data and performed data imputation. We utilized the potently accurate and efficient Support Vector Regression (SVR) algorithm.

## Challenges we ran into
We ran into the massive challenge of trying to integrate our PyTorch mobile backend with our flutter front-end. We tried many ways to get around this such as using ONNX and converting the PyTorch model into a TensorFlow model, but had no luck. Although we were ultimately unsuccessful, we are grateful for all of new concepts we learned along the way.


## Accomplishments that we're proud of
We are incredibly proud of our stunning website and app front-end as well as the efficiency and accuracy of our AI models. We are also very proud of being able to finish everything on time. 

## What we learned
We learned that data collection and analysis take up significantly more time than building and fitting a model to the data itself. We also learned that the integration of frontend and backend components is an extremely complex process and to plan accordingly. 

## What's next for Recyclify
We mainly plan on expanding the variety of our models available to both individuals and governments.
We hope to ultimately streamline Recyclify organic material and composting developments as well as usage of object detection algorithms to further support the cause of environmentally conscious waste disposal. We would also like to create a backend database connecting the app and the website together. Lastly, we plan to expand upon the functionality our mobile application.

### Citations and image sources
https://www.unep.org/interactive/beat-plastic-pollution/#:~:text=Only%209%25%20of%20all%20plastic,dumps%20or%20the%20natural%20environment.

https://www.metalmenrecycling.com.au/what-happens-if-we-dont-recycle/

https://glescrap.com/the-effects-of-not-recycling/

https://www.recyclingbins.co.uk/blog/environmental-impacts-of-recycling/

https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.farfetchtechblog.com%2Fen%2Fblog%2Fpost%2Fhow-to-be-accessible%2F&psig=AOvVaw0h06UN3EuawtZbCxx55lPl&ust=1630930112735000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCLCJs6Pm5_ICFQAAAAAdAAAAABAD

https://www.google.com/imgres?imgurl=https%3A%2F%2Fefficiencyforall.org%2Fwordpress%2Fwp-content%2Fuploads%2F2018%2F12%2FEfficiencyIsEfficientSM.jpg&imgrefurl=https%3A%2F%2Fefficiencyforall.org%2Fwordpress%2F2018%2F12%2F18%2Fefficiency-is-efficient%2F&tbnid=HHy8MlSsHnjrmM&vet=12ahUKEwiKuOS_5ufyAhVLh54KHb9JChEQMygBegUIARDNAQ..i&docid=nLqaQ1IMaqs_XM&w=750&h=450&itg=1&q=efficient&ved=2ahUKEwiKuOS_5ufyAhVLh54KHb9JChEQMygBegUIARDNAQ

https://www.google.com/imgres?imgurl=https%3A%2F%2Fwww.thesimpledollar.com%2Fwp-content%2Fuploads%2F2020%2F04%2FTheSimpleDollar-Fun-With-Friends.png&imgrefurl=https%3A%2F%2Fwww.thesimpledollar.com%2Fsave-money%2Fcheap-and-social-15-inexpensive-and-very-fun-things-to-do-with-friends%2F&tbnid=gHwrwJEym9ugVM&vet=12ahUKEwjqmJ3K5ufyAhUFyDgKHQjcDbMQMygCegUIARDOAQ..i&docid=TndXzGjYKeRZhM&w=5209&h=2362&itg=1&q=fun&ved=2ahUKEwjqmJ3K5ufyAhUFyDgKHQjcDbMQMygCegUIARDOAQ

https://miro.medium.com/max/900/1*TBpm9SCsNZ1ovLS1FzdsRg.jpeg

https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Conv_layers.png/474px-Conv_layers.png

https://www.google.com/imgres?imgurl=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1400%2F1*7oukapIBInsovpHkQB3QZg.jpeg&imgrefurl=https%3A%2F%2Ftowardsdatascience.com%2Fcheat-sheet-for-google-colab-63853778c093&tbnid=eRY5QcmBPoy2WM&vet=12ahUKEwjFhvyj8OfyAhWCkp4KHetPBXoQMygBegUIARCtAQ..i&docid=7o3a0NktrrJCwM&w=1400&h=619&itg=1&q=google%20colab&client=safari&ved=2ahUKEwjFhvyj8OfyAhWCkp4KHetPBXoQMygBegUIARCtAQ

https://iotsimplified.com/2020/05/15/image-object-detection-sorting-to-assist-plastic-recycling-facilities/

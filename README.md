# flower_identifer
This can predict the name of flowers

# What it contains
It contains two files, train.py and predict.py.  
train.py for training your network  
predict.py for predicting flower  
flower dataset will be uploaded if possible

# Using predict.py
You can simply run this file by adding a parameter --path path/to/your/image.jpg  
By default it:  
prints out top 3 possibilities, uses gpu, loads checkpoint as 'checkpoint.pth',
loads json file 'cat_to_name.json' which contains names for indexes.
But you can add optional parameters to modify these values.

# Optional Parameters for predict.py
```
--path path/image.jpg (path to flower image)

--gpu True OR False (use gpu or not)

--top_k 5 (number of possibilities to be printed)

--checkpoint path/checkpoint.pth (path to saved checkpoint)

--cat_json path/category.json (path to json contains names for indexes)
```

# Using train.py
You can simply run this file by using parameter --data_dir path/to/data_directory  
By default it:  
uses model VGG16, uses gpu, learning rate 0.001, epochs 8, hidden layers 2,
saves checkpoint as 'checkpoint.pth'
But you can add optional paramters to modify these values

# Optional Parameters for train.py
```
--gpu True OR False

--arch model (choose either VGG16 OR Densenet121)

--lr 0.005 (specify learning rate)

--epochs 10 (specify number of epochs)

--hidden_units 256 (number of in_features for 2nd hidden layer)

--saved_model path/checkpoint.pth (locate where to save checkpoint)
```

# Thanks for using
I made this project to sharpen my skills.  
Feel free to modify, I'll merge any good changes

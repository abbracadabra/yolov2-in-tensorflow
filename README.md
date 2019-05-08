## about model
- implementation of yolov2 concepts,but not 100% in accordance with every detail
- vgg16 as image encoder
- trained on voc2012 for 2 epochs
- 7\*7 total cells in detection layer,5 box predictors in each cell,total detection layer size is 7\*7\*(5\*5+20)

## about code
- train.py #train
- evaluate.py #predict
- config.py #config

## some random non cherry-picked test examples
<div>
<img src='https://user-images.githubusercontent.com/35487258/57367790-4a8c2680-71bc-11e9-9f0e-28b6942e9339.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367791-4b24bd00-71bc-11e9-9d92-5f8b31478f1e.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367792-4b24bd00-71bc-11e9-83ee-38595afb08eb.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367793-4bbd5380-71bc-11e9-8e2d-4e2b003d47d4.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367795-4c55ea00-71bc-11e9-897e-bc8965ea40b1.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367798-4cee8080-71bc-11e9-8ca0-a1a0406436a8.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367799-4cee8080-71bc-11e9-81ec-134f5663260a.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367800-4e1fad80-71bc-11e9-9c54-ff38804fffec.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367801-4e1fad80-71bc-11e9-85be-92da94a388f9.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367802-4eb84400-71bc-11e9-9ec0-da69c0f69d81.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367803-4f50da80-71bc-11e9-8358-433ae7b1b031.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367804-4f50da80-71bc-11e9-866c-7c2425a09a59.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367806-4fe97100-71bc-11e9-8909-af9cbeafe6b0.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367809-50820780-71bc-11e9-84fa-129f89d406e8.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367810-50820780-71bc-11e9-91d2-bd76369211af.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367811-511a9e00-71bc-11e9-8bfa-f19a019872df.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367813-511a9e00-71bc-11e9-90cc-db200070cd2c.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367818-51b33480-71bc-11e9-9bb6-9b5a1ca96d7a.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367820-524bcb00-71bc-11e9-8629-6d0069ab906e.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367821-52e46180-71bc-11e9-974f-49ac361e1518.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367822-52e46180-71bc-11e9-8707-47240cff2ca9.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367825-537cf800-71bc-11e9-9e42-a4c99440b111.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367831-5546bb80-71bc-11e9-9465-cfd044e0be71.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367834-5546bb80-71bc-11e9-91bc-059b8687c97d.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367835-55df5200-71bc-11e9-9557-79df9d799a5a.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367836-5677e880-71bc-11e9-934e-b29a1d914516.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367838-5677e880-71bc-11e9-9671-7d9ed391aec1.jpg'>
<img src='https://user-images.githubusercontent.com/35487258/57367839-57a91580-71bc-11e9-9a06-29a799e7cbf0.jpg'>
</div>

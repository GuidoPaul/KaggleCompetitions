#https://www.kaggle.com/product-feedback/21591
#1 chrome install extension  https://chrome.google.com/webstore/detail/cookiestxt/njabckikapfpffapmjgojcnbfjonfjfg
#2 login kaggle
#3 export cookie.txt [# To download cookies for this tab click here, or download all cookies.]
#4 run this shell

wget -x -c --load-cookies cookies.txt -P input -nH --cut-dirs=5 https://www.kaggle.com/c/dog-breed-identification/download/labels.csv.zip

wget -x -c --load-cookies cookies.txt -P input -nH --cut-dirs=5 https://www.kaggle.com/c/dog-breed-identification/download/sample_submission.csv.zip

wget -x -c --load-cookies cookies.txt -P input -nH --cut-dirs=5 https://www.kaggle.com/c/dog-breed-identification/download/test.zip

wget -x -c --load-cookies cookies.txt -P input -nH --cut-dirs=5 https://www.kaggle.com/c/dog-breed-identification/download/train.zip

cd ./input
unzip -a \*.zip
rm -f labels.csv.zip sample_submission.csv.zip test.zip train.zip


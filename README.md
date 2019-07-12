# Hopfield
## 程式執行說明
點擊Hopfield.exe檔，稍微等待3分鐘後會出現Hopfield執行介面
可直接點選上方Select dataset下拉式選單，選擇要作為預測的data的file名稱。
點選後下方會直接出現三張圖，若是選basic data就是dataset裡面的所有資料，若是Bonus則是前三張圖片。

選完dataset後，按start training則會開始計算Hopfield的W以及將testing data丟進來做預測回想(prediction recall)，最下方顯示這個圖片是經過幾次疊代後的結果。
右上角則會顯示整個dataset的預測accuracy。

若是Bonus 的資料集，因為有15個sample，所以我分批顯示，再選完Bonus時，會先顯示前三個sample
按完start training後，就可以一直按Next 3 Sample，來看後面的data以及他的預測結果。
 
## Interface
![image](https://github.com/a83117700/Hopfield/blob/master/Interface.png)  


Oliver Yao
a83117700@gmail.com


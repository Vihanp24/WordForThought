{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'sklearn'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 3\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mpandas\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mpd\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mmatplotlib\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpyplot\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mplt\u001b[39;00m\n\u001b[1;32m----> 3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mlinear_model\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m LinearRegression\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodel_selection\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m train_test_split\n\u001b[0;32m      5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmetrics\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m mean_squared_error\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'sklearn'"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.metrics import r2_score\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "df=pd.read_csv('Housing.csv')\n",
    "\n",
    "#we have many data which come under classification class.we cannont directly use them for training and testing the ML.\n",
    "#therefore we first convert them into binary{0,1} using a method LabelEncoder(), which i found from the website (https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.LabelEncoder.html)\n",
    "le = LabelEncoder()\n",
    "df['mainroad'] = le.fit_transform(df['mainroad'])\n",
    "df['guestroom'] = le.fit_transform(df['guestroom'])\n",
    "df['basement'] = le.fit_transform(df['basement'])\n",
    "df['hotwaterheating'] = le.fit_transform(df['hotwaterheating'])\n",
    "df['airconditioning'] = le.fit_transform(df['airconditioning'])\n",
    "df['prefarea'] = le.fit_transform(df['prefarea'])\n",
    "df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])\n",
    "\n",
    "#now we seperate the regressor and and the response\n",
    "X=df.iloc[:,1:]\n",
    "y=df.iloc[:,0]\n",
    "\n",
    "#now i split the date into training and testing set\n",
    "Xtrain, Xtest, ytrain, ytest=train_test_split(X, y, test_size = 0.3, random_state=0)\n",
    "\n",
    "\n",
    "#Now I will trian the data.\n",
    "reg= LinearRegression()\n",
    "reg.fit(Xtrain, ytrain)\n",
    "\n",
    "ytrainpredict=reg.predict(Xtrain)\n",
    "mse= mean_squared_error(ytrain, ytrainpredict)\n",
    "r2= r2_score(ytrain, ytrainpredict)\n",
    "\n",
    "#we will print the mse and r2 values for both training set and testing set.\n",
    "print('Train MSE =', mse)\n",
    "print('Train R2 score =', r2)\n",
    "print(\"\\n\")\n",
    "\n",
    "ytestpredict = reg.predict(Xtest)\n",
    "mse= mean_squared_error(ytest, ytestpredict)\n",
    "r2= r2_score(ytest, ytestpredict)\n",
    "\n",
    "print('Train MSE =', mse)\n",
    "print('Train R2 score =', r2)\n",
    "print(\"\\n\")\n",
    "\n",
    "#we are printing the bias and the respecive slopes for each regressors\n",
    "slope= reg.coef_\n",
    "bias= reg.intercept_\n",
    "print(slope, bias)\n",
    "\n",
    "#we will be plotting ytest and ytestpredict and compare with the actual vs predicted line.\n",
    "plt.figure()\n",
    "plt.scatter(ytest, ytestpredict, color='blue', alpha=0.6)\n",
    "plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red', linestyle='--')\n",
    "plt.title('Actual vs Predicted Values') \n",
    "plt.xlabel('Actual values')\n",
    "plt.ylabel('Predicted values')\n",
    "plt.grid()\n",
    "plt.show()\n",
    "    \n",
    "    \n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

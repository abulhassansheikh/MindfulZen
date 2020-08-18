from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == 'POST':
        
        os.chdir("C:/Users/abulh/Sync/Projects/Health&WellbeingApp/API/app")

        StressRef = pd.read_csv("C:/Users/abulh/Sync/Projects/Health&WellbeingApp/API/app/StressRef.csv")
        RecomText = pd.read_csv("C:/Users/abulh/Sync/Projects/Health&WellbeingApp/API/app/RecomText.csv")
        
        #Evaluate Different stress levels
        def factorAnalysis(predictStress, factorlist, ActualAnswers, StressRefSS, model, FactorDirection):
         
            LongrunDecStress_Ideal = round(predictStress-1)
            LRDdf = pd.DataFrame({"factors":factorlist, "Actual":ActualAnswers[0:len(factorlist)], "avg":0, "CurrStress":predictStress,"newstress":0})
             
            for f in range(0, len(factorlist)):
                factor = f
                factorlabel = factorlist[factor]
                factorAvgValue = StressRefSS[StressRefSS["DAILY_STRESS"] == LongrunDecStress_Ideal].reset_index()[factorlabel][0]
                LRDvalDF = pd.DataFrame(ActualAnswers)
                LRDvalDF.iloc[factor, 0] = factorAvgValue
                newStress = model.predict(np.array(LRDvalDF[0]).reshape(1, -1))[0]
                 
                LRDdf.loc[LRDdf["factors"] == factorlabel, "avg"]= factorAvgValue
                LRDdf.loc[LRDdf["factors"] == factorlabel, "newstress"]= newStress
             
            LRDdf["ActivityChange"] = LRDdf["avg"].astype(float) - LRDdf["Actual"].astype(float) #increase behaviour 
            LRDdf["stressChange"] = LRDdf["newstress"] - LRDdf["CurrStress"] #more neg, the better
            LRDdf = LRDdf[["factors", "ActivityChange", "stressChange"]]
             
            if FactorDirection == "increase":
                LRDdf = LRDdf[(LRDdf["ActivityChange"]>0) & (LRDdf["stressChange"]<0.001)].sort_values("stressChange")
            elif FactorDirection == "reduce":
                LRDdf = LRDdf[(LRDdf["ActivityChange"]<0) & (LRDdf["stressChange"]<0.001)].sort_values("stressChange")
                 
            return LRDdf
        
        # load the model from disk
        modelLongInc = pickle.load(open("longRunStressIncModel.pkl", 'rb'))
        modelLongDec = pickle.load(open("longRunStressDecModel.pkl", 'rb'))
        modelShortInc = pickle.load(open("ShortRunStressIncModel.pkl", 'rb'))
        modelShortDec = pickle.load(open("ShortRunStressDecModel.pkl", 'rb'))
       
        PLACES_VISITED = request.form["PLACES_VISITED"]
        ACHIEVEMENT = request.form["ACHIEVEMENT"]
        BMI_RANGE = request.form["BMI_RANGE"]
        LIVE_VISION = request.form["LIVE_VISION"]
        LOST_VACATION = request.form["LOST_VACATION"]
        PERSONAL_AWARDS = request.form["PERSONAL_AWARDS"]
        CORE_CIRCLE = request.form["CORE_CIRCLE"]
        SUFFICIENT_INCOME = request.form["SUFFICIENT_INCOME"]
        DONATION = request.form["DONATION"]
        AGE = request.form["AGE"]
        GENDER = request.form["GENDER"]
        FRUITS_VEGGIES= request.form["FRUITS_VEGGIES"]
        SUPPORTING_OTHERS= request.form["SUPPORTING_OTHERS"]
        SOCIAL_NETWORK= request.form["SOCIAL_NETWORK"]
        TODO_COMPLETED= request.form["TODO_COMPLETED"]
        FLOW= request.form["FLOW"]
        DAILY_STEPS= request.form["DAILY_STEPS"]
        SLEEP_HOURS= request.form["SLEEP_HOURS"]
        DAILY_SHOUTING= request.form["DAILY_SHOUTING"]
        TIME_FOR_PASSION= request.form["TIME_FOR_PASSION"]
        DAILY_MEDITATION = request.form["DAILY_MEDITATION"] 

        
        #Long run recommendation
        longRunsAnswersInc = [PLACES_VISITED, 
                                    ACHIEVEMENT, 
                                    LIVE_VISION, 
                                    PERSONAL_AWARDS, 
                                    CORE_CIRCLE, 
                                    SUFFICIENT_INCOME, 
                                    DONATION, 
                                    AGE, 
                                    GENDER]
        # Long run recommendation
        longRunsAnswersDec = [BMI_RANGE, 
                                LOST_VACATION, 
                                AGE, 
                                GENDER]
        #Long run recommendation
        ShortRunsAnswersInc = [FRUITS_VEGGIES, 
                                    SUPPORTING_OTHERS, 
                                    SOCIAL_NETWORK, 
                                    TODO_COMPLETED, 
                                    FLOW, 
                                    DAILY_STEPS, 
                                    SLEEP_HOURS, 
                                    TIME_FOR_PASSION, 
                                    DAILY_MEDITATION,
                                    AGE, 
                                    GENDER]
        #Short run recommendation
        ShortRunsAnswersDec = [DAILY_SHOUTING, 
                                AGE, 
                                GENDER]
        
        longrunIncStress = modelLongInc.predict(np.array(longRunsAnswersInc).reshape(1, -1))[0]
        longrunDecStress = modelLongDec.predict(np.array(longRunsAnswersDec).reshape(1, -1))[0]
        ShortrunIncStress = modelShortInc.predict(np.array(ShortRunsAnswersInc).reshape(1, -1))[0]
        ShortrunDecStress = modelShortDec.predict(np.array(ShortRunsAnswersDec).reshape(1, -1))[0]
        
        StressRefSS = StressRef.loc[StressRef["AGE"].isin([AGE])]
        StressRefSS = StressRefSS.loc[StressRefSS["GENDER"].isin([GENDER])]
       
        #Long Run & Increaseing
        LRIdf = (factorAnalysis(predictStress = longrunIncStress,
                        factorlist = ['PLACES_VISITED', 
                                    'ACHIEVEMENT',
                                    'LIVE_VISION',
                                    'PERSONAL_AWARDS', 
                                    'CORE_CIRCLE',
                                    'SUFFICIENT_INCOME',
                                    'DONATION'],
                        ActualAnswers = longRunsAnswersInc,
                        StressRefSS = StressRefSS,
                        model = modelLongInc, 
                        FactorDirection = "increase"))
           
        #Short Run & Increaseing
        SRIdf = (factorAnalysis(predictStress = ShortrunIncStress,
                        factorlist = ["FRUITS_VEGGIES", 
                                    'SUPPORTING_OTHERS', 
                                    'SOCIAL_NETWORK', #Daily interaction with others?
                                    'TODO_COMPLETED', 
                                    'FLOW', 
                                    'DAILY_STEPS', 
                                    'SLEEP_HOURS', 
                                    'TIME_FOR_PASSION', 
                                    "DAILY_MEDITATION"],
                        ActualAnswers = ShortRunsAnswersInc,
                        StressRefSS = StressRefSS,
                        model = modelShortInc, 
                        FactorDirection = "increase"))
        
        #Long Run & Decreasing
        LRDdf = (factorAnalysis(predictStress = longrunDecStress,
                        factorlist = ["BMI_RANGE", "LOST_VACATION"],
                        ActualAnswers = longRunsAnswersDec,
                        StressRefSS = StressRefSS,
                        model = modelLongDec, 
                        FactorDirection = "reduce"))
              
        #Short Run & Decreasing
        SRDdf = (factorAnalysis(predictStress = ShortrunDecStress,
                        factorlist = ["DAILY_SHOUTING"],
                        ActualAnswers = ShortRunsAnswersDec,
                        StressRefSS = StressRefSS,
                        model = modelShortDec, 
                        FactorDirection = "reduce"))
        
        #Analyse results and provide recommendations
        #decreasing factors
        finalrecomdf = LRDdf
        finalrecomdf = LRIdf.append(SRIdf)
        finalrecomdf = finalrecomdf.append(LRDdf)
        finalrecomdf = finalrecomdf.append(SRDdf)
        
        print("===========")
        finalrecomdf = finalrecomdf.sort_values("stressChange").reset_index()
        print(finalrecomdf)
        #Output   
        recomOne = finalrecomdf["factors"][0]
        recomOneSize = abs(round(finalrecomdf["ActivityChange"][0]))
        recomOnedir = RecomText[RecomText['factors'] == recomOne].reset_index()["FactorDirection"][0]
        recomOnedec = RecomText[RecomText['factors'] == recomOne].reset_index()["description"][0]
        recomstringOne = str(recomOnedir)+" "+str(recomOnedec)+" "+str(recomOneSize)
        
        recomTwo = finalrecomdf["factors"][1]
        recomTwoSize = abs(round(finalrecomdf["ActivityChange"][1]))
        recomTwodir = RecomText[RecomText['factors'] == recomTwo].reset_index()["FactorDirection"][0]
        recomTwodec = RecomText[RecomText['factors'] == recomTwo].reset_index()["description"][0]
        recomstringTwo = str(recomTwodir)+" "+str(recomTwodec)+" "+str(recomTwoSize)
        
        recomThree = finalrecomdf["factors"][2]
        recomThreeSize = abs(round(finalrecomdf["ActivityChange"][3]))
        recomThreedir = RecomText[RecomText['factors'] == recomThree].reset_index()["FactorDirection"][0]
        recomThreedec = RecomText[RecomText['factors'] == recomThree].reset_index()["description"][0]
        recomstringThree = str(recomThreedir)+" "+str(recomThreedec)+" "+str(recomThreeSize)
        
        
        FinalRecommendationString = ("We find that you can reduce your stress level and improve "+
                                     "the quality of your life substantially by "+recomstringOne+
                                      ". Additionally, we find you could benefit by "+recomstringTwo+ 
                                      " and, "+recomstringThree+".")
       
        Data = [["Recommendations", FinalRecommendationString]]
        
        return render_template('index.html',
                               Data=Data)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)


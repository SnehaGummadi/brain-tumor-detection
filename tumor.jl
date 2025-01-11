using CMPUtils, CSV, DataFrames, ScikitLearn, PyCall, JLD, PyCallJLD, PyPlot

# to split data for training and testing
@sk_import model_selection: train_test_split

# To implement a shallow neural network
@sk_import neural_network: MLPClassifier

# To implement a logistic regression
@sk_import linear_model: LogisticRegression

# Functions to determine the effectiveness of the models
@sk_import metrics: accuracy_score
@sk_import metrics: confusion_matrix
@sk_import metrics: ConfusionMatrixDisplay
@sk_import metrics: classification_report
@sk_import model_selection: KFold
@sk_import model_selection: cross_validate

# Get pathways for nontumorous and tumorous files
noTumorPath = joinpath(pwd(), "tumorsimage", "no")
tumorPath = joinpath(pwd(), "tumorsimage", "yes")

# Recode images with 60 features
recodeimage(noTumorPath)
recodeimage(tumorPath)

# get path for recoded data's csv files
noTumorcsv = joinpath(noTumorPath, "image_recoded.csv")
tumorcsv = joinpath(tumorPath, "image_recoded.csv")

# Read from csv files into variables
nontumorous = CSV.read(noTumorcsv, DataFrame)
tumorous = CSV.read(tumorcsv, DataFrame)

# Determine what each option will represent what numerical value
nontumorous.target .= 1
tumorous.target .= 0

# Place all data into one variable
fulldata = vcat(nontumorous, tumorous)

# Determine the features and target
features = select(fulldata, Not(:target))|> Array
target = select(fulldata, :target)|> Array

# Split 70% of the data to train the models and 30% to test the effectiveness of the models
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3,random_state = 342)

# implement shallow neural network
shallow_nn = MLPClassifier(hidden_layer_sizes = (100,100))

# Train shallow neural network
fit!(shallow_nn, features_train, target_train)

# Ask shallow neural network to predict from the test set
shallow_nn_predictions = predict(shallow_nn, features_test)

accuracy_score(target_test, shallow_nn_predictions)
shallowCF = confusion_matrix(target_test, shallow_nn_predictions)
disp= ConfusionMatrixDisplay(confusion_matrix = shallowCF)
disp.plot()
gcf()

print(classification_report(target_test, shallow_nn_predictions))
cvShallowNN = cross_validate(shallow_nn,
                        features_train, target_train,
                        cv = KFold(5),
                        return_estimator = true,
                        return_train_score = true,
                        scoring = ["accuracy",
                                        "recall_weighted",
                                        "precision_weighted"])

cv_df = DataFrame(cvShallowNN)[!, Not([:estimator, :fit_time, :score_time])]
mean_performance = describe(cv_df)[!, [:variable, :mean]]


# Save the shallow neural network model
save("Tumor_Detector.jld", "shallow_model", shallow_nn)


# DOES THE FOLLOWING HAVE TO BE HERE?????
# Load model so that it does not have to be trained repeatedly
#mSNN = load("Tumor_Detector.jld", "shallow_model")

#implement logistic regression
simple_logistic = LogisticRegression()

# Train the logistic regression model with the training set
fit!(simple_logistic, features_train, target_train)

# Save the logistic regression model
save("Tumor_Detector.jld", "logistic_model", simple_logistic)

# Load model so that it does not have to be trained repeatedly
mLog = load("Tumor_Detector.jld", "logistic_model")

# Get the logistic regression's predictions
logistic_regression_predictions = predict(mLog, features_test)

# Create empty canvas to put plots on
figure()

# Get the accuracy score of how well the neural network performed on the test data set
accuracy_score(target_test, shallow_nn_predictions)

# Get accuracy score for logistic regression
accuracy_score(target_test,logistic_regression_predictions )

# Confusion matrix for shallow neural network
shallowCF = confusion_matrix(target_test, shallow_nn_predictions)

# Confusion matrix for logisitic regression
logisticCF = confusion_matrix(target_test, logistic_regression_predictions)

# how to display confusion matrix for shallow neural network


# Display confusion matrix for logisitic regression
disp= ConfusionMatrixDisplay(confusion_matrix = logisticCF, display_labels = simple_logistic.classes_)
disp.plot()
gcf()

print(classification_report(target_test, logistic_regression_predictions))

# Display confusion matrix for shallow neural network
disp= ConfusionMatrixDisplay(confusion_matrix = shallowCF)
disp.plot()
gcf()

print(classification_report(target_test, shallow_nn_predictions))

cvLogistic = cross_validate(simple_logistic,
                        features_train, target_train,
                        cv = KFold(5),
                        return_estimator = true,
                        return_train_score = true,
                        scoring = ["accuracy",
                                        "recall_weighted",
                                        "precision_weighted"])

cv_df = DataFrame(cvLogistic)[!, Not([:estimator, :fit_time, :score_time])]
mean_performance = describe(cv_df)[!, [:variable, :mean]]


cvShallowNN = cross_validate(shallow_nn,
                        features_train, target_train,
                        cv = KFold(5),
                        return_estimator = true,
                        return_train_score = true,
                        scoring = ["accuracy",
                                        "recall_weighted",
                                        "precision_weighted"])

cv_df = DataFrame(cvShallowNN)[!, Not([:estimator, :fit_time, :score_time])]
mean_performance = describe(cv_df)[!, [:variable, :mean]]


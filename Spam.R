# Author:         Weijia Xiong (xwjanthony@gmail.com)
# Purpose:        Extract features/keywords from text files and generate a frequency matrix;
#                 Applied several types of machine learning algorithms to classify spam from ham;
#                 Test the accuracy/APER of different models.
# Data Used:      Email messages from Gmail inboxes of contributors
#                 Total:624  Ham:205  Spam:419
# Packages Used:  caret, data.table, klaR, rattle, rpart, randomForest, tm
#                 (including their dependent packages)


# Acknowledgement: Drew Conway (the author of "Machine Learning for Hackers")

# NOTE: If you are running this in the R console you must use the 'setwd' command to set the 
# working directory for the console to whereever you have saved this file prior to running.
# Otherwise you will see errors when loading data or saving figures!

#****************************************************************************************

# Load libraries
# Make sure all packages are already installed!
library(caret)
library(data.table)
library(klaR)
library(rattle)
library(rpart)
library(randomForest)
library(tm)

#****************************************************************************************

# Load custom functions:

# Return a single element vector of email body.
# This is a very simple approach, as we are only using words as features.
get.msg <- function(path){
  con <- file(path, open = "rt", encoding = "latin1")
  text <- readLines(con)
  close(con)
  return(paste(text, collapse = "\n"))
}

# Create a Term Document Matrix (TDM) from the corpus of email.
# This TDM is used to create the feature set used for training classifier.
get.tdm <- function(doc.vec){
  control <- list(stopwords = TRUE,
                  removePunctuation = TRUE,
                  removeNumbers = TRUE,
                  minDocFreq = 2)
  doc.corpus <- Corpus(VectorSource(doc.vec))
  doc.dtm <- TermDocumentMatrix(doc.corpus, control)
  return(doc.dtm)
}

# Extract specific features from TDM,
# including the frequency of typical words, capital letters, numbers and dollar sign.
FeatureMatrix <- function(doc.vec){
  tdm <- get.tdm(doc.vec)
  fmatrix <- as.matrix(tdm)
  # calculate frequency
  for(i in 1:ncol(fmatrix))
    fmatrix[,i] <- 100*fmatrix[,i]/colSums(fmatrix)[i]
  # count uppercase & lower-case letters, numbers and dollar sign
  num.upper <- sapply(doc.vec,
                      function (p) sum(between(charToRaw(p), charToRaw("A"), charToRaw("Z"))))
  num.lower <- sapply(doc.vec,
                      function (p) sum(between(charToRaw(p), charToRaw("a"), charToRaw("z"))))
  num.number <- sapply(doc.vec,
                       function (p) sum(between(charToRaw(p), charToRaw("0"), charToRaw("9"))))
  num.dollar <- sapply(doc.vec,
                       function (p) sum(between(charToRaw(p), charToRaw("$"), charToRaw("$"))))
  # calculate frequency
  Capital <- 100*num.upper/(num.upper+num.lower)
  Number <- 100*num.number/(num.upper+num.lower+num.number+num.dollar)
  Dollar <- 100*num.dollar/(num.upper+num.lower+num.number+num.dollar)
  df <- data.frame(cbind(t(fmatrix),Capital,Number,Dollar))
  # choose typical words in spam emails
  df <- df[,which(names(df)=="bonus"
                  | names(df)=="cash"
                  | names(df)=="deal"
                  | names(df)=="discount"
                  | names(df)=="earn"
                  | names(df)=="free"
                  | names(df)=="gift"
                  | names(df)=="member"
                  | names(df)=="off"
                  | names(df)=="offer"
                  | names(df)=="order"
                  | names(df)=="pay"
                  | names(df)=="price"
                  | names(df)=="profit"
                  | names(df)=="sale"
                  | names(df)=="save"
                  | names(df)=="unsubscribe"
                  | names(df)=="Capital"
                  | names(df)=="Number"
                  | names(df)=="Dollar"),drop=F]
  df <- na.omit(df)  # remove NA rows
  return(df)
}

# Select united features from both spam and ham.
EmailMatrix <- function(x,y){
  union <- union(names(x),names(y))  # intersect set
  diffx <- setdiff(union,names(x))  # difference set of x from y
  diffy <- setdiff(union,names(y))  # difference set of y from x
  # create zero matrix if words don't exist
  x1 <- as.data.frame(matrix(0,nrow(x),length(diffx)))
  names(x1) <- diffx
  y1 <- as.data.frame(matrix(0,nrow(y),length(diffy)))
  names(y1) <- diffy
  # intergrate new matrices
  xnew <- cbind(x,x1)
  ynew <- cbind(y,y1)
  email <- rbind(xnew,ynew)
  return(email)
}

#****************************************************************************************

# Extract Features:

# read paths of text files
spam.path<-file.path("spam")
ham.path<-file.path("ham")

# read documents
spam.docs <- dir(spam.path)
ham.docs <- dir(ham.path)

# get email body into a single vector
all.spam <- sapply(spam.docs,function(p) get.msg(file.path(spam.path, p)))
all.ham <- sapply(ham.docs,function(p) get.msg(file.path(ham.path, p)))

# extract features and combine into a single matrix
spam <- FeatureMatrix(all.spam)
ham <- FeatureMatrix(all.ham)
type <- c(rep("spam",dim(spam)[1]),rep("ham",dim(ham)[1]))
email <- cbind(EmailMatrix(spam,ham),type)
View(email)

#****************************************************************************************

# Test classification models:

# split into training and testing datasets
set.seed(3)
inTrain<-createDataPartition(y=email$type,p=0.7,list=F)
training<-email[inTrain,]
testing<-email[-inTrain,]

# linear discriminant analysis (simplest)
# APER = 39/186 = 20.97%
mod.lda <- train(type~.,data=training,method="lda")
pred.lda <- predict(mod.lda,testing)
table(pred.lda,testing$type)

# naive bayes
# APER = 63/186 = 33.87%
# Maybe due to large amount of 0 in datasets
mod.nb <- train(type~.,data=training,method="nb")
pred.nb <- predict(mod.nb,testing)
table(pred.nb,testing$type)

# decision trees
# APER = 27/186 = 14.52%
mod.tree <- train(type~.,method="rpart",data=training)
fancyRpartPlot(mod.tree$finalModel)  # plot decision tree 
pred.tree <- predict(mod.tree,testing)
table(pred.tree,testing$type)

# random forest (most accurate)
# APER = 13/186 = 7.03%
mod.rf <- train(type~.,data=training,method="rf",prox=T)
pred.rf <- predict(mod.rf,testing)
table(pred.rf,testing$type)

# k-means clustering (unsupervised learning)
kMeans <- kmeans(subset(training,select=-c(type)),centers=2) # clustering
training$clusters <- as.factor(kMeans$cluster)
mod.km <- train(clusters~.,data=subset(training,select=-c(type)),method="rpart")
pred.km <-predict(mod.km,testing)
table(pred.km,testing$type)
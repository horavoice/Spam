library(caret)
library(data.table)
library(klaR)
library(rattle)
library(rpart)
library(randomForest)
library(tm)

# Return a single element vector of email body.
get.msg <- function(path){
  con <- file(path, open = "rt", encoding = "latin1")
  text <- readLines(con)
  close(con)
  return(paste(text, collapse = "\n"))
}

# Create a Term Document Matrix (TDM) from the corpus of email.
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
  for(i in 1:ncol(fmatrix))
    fmatrix[,i] <- 100*fmatrix[,i]/colSums(fmatrix)[i]
  #count uppercase & lower-case letters, numbers and dollar sign
  num.upper <- sapply(doc.vec,
                      function (p) sum(between(charToRaw(p), charToRaw("A"), charToRaw("Z"))))
  num.lower <- sapply(doc.vec,
                      function (p) sum(between(charToRaw(p), charToRaw("a"), charToRaw("z"))))
  num.number <- sapply(doc.vec,
                       function (p) sum(between(charToRaw(p), charToRaw("0"), charToRaw("9"))))
  num.dollar <- sapply(doc.vec,
                       function (p) sum(between(charToRaw(p), charToRaw("$"), charToRaw("$"))))
  #calculate frequency
  Capital <- 100*num.upper/(num.upper+num.lower)
  Number <- 100*num.number/(num.upper+num.lower+num.number+num.dollar)
  Dollar <- 100*num.dollar/(num.upper+num.lower+num.number+num.dollar)
  df <- data.frame(cbind(t(fmatrix),Capital,Number,Dollar))
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
  df <- na.omit(df) #remove NA rows
  return(df)
}

# Select united features from both SPAM and HAM.
EmailMatrix <- function(x,y){
  union <- union(names(x),names(y))
  diffx <- setdiff(union,names(x))
  diffy <- setdiff(union,names(y))
  x1 <- as.data.frame(matrix(0,nrow(x),length(diffx)))
  names(x1) <- diffx
  y1 <- as.data.frame(matrix(0,nrow(y),length(diffy)))
  names(y1) <- diffy
  xnew <- cbind(x,x1)
  ynew <- cbind(y,y1)
  email <- rbind(xnew,ynew)
  return(email)
}

#****************************************************************************************

# read paths of text files
spam.path<-file.path("~/Dropbox/508 Projects in Global Operations Management/text/spam/")
ham.path<-file.path("~/Dropbox/508 Projects in Global Operations Management/text/ham/")

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

#split into training and testing datasets
set.seed(3)
inTrain<-createDataPartition(y=email$type,p=0.7,list=F)
training<-email[inTrain,]
testing<-email[-inTrain,]

#linear discriminant analysis
mod.lda <- train(type~.,data=training,method="lda")
pred.lda <- predict(mod.lda,testing)
table(pred.lda,testing$type)

#naive bayes
mod.nb <- train(type~.,data=training,method="nb")
pred.nb <- predict(mod.nb,testing)
table(pred.nb,testing$type)

#decision trees
mod.tree <- train(type~.,method="rpart",data=training)
fancyRpartPlot(mod.tree$finalModel) #plot tree 
pred.tree <- predict(mod.tree,testing)
table(pred.tree,testing$type)

#random forest
mod.rf <- train(type~.,data=training,method="rf",prox=T)
pred.rf <- predict(mod.rf,testing)
table(pred.rf,testing$type)

#k-means clustering (unsupervised)
kMeans <- kmeans(subset(training,select=-c(type)),centers=2) #clustering
training$clusters <- as.factor(kMeans$cluster)
mod.km <- train(clusters~.,data=subset(training,select=-c(type)),method="rpart")
pred.km <-predict(mod.km,testing)
table(pred.km,testing$type)
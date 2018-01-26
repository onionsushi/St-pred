library(quantmod)
stock <- read.table(file = "Desktop/stocklist.txt", sep = "\t", quote = "")
stocknames <- stock$V2
stocknames <- gsub(x = stocknames, pattern = " ", replacement = "")
download_stockdata <- function(){
  for (i in (1:5)){
    print(i)
    getSymbols(as.character(stocknames[i]))
    Sys.sleep(5)
  }
}
download_stockdata()
getSymbols(as.character(stocknames[2]))

{
  for (i in (50:100)){
    print(i)
    getSymbols(as.character(stocknames[i]))
    Sys.sleep(5)
  }
}
stocknames[49]
functioniongnumber <- c(1:23, 25:48, 50:100)

refinestock <- function(stockname){
  pricechart <- as.numeric(stockname[,4])
  chartlength <- length(pricechart)
  pricechangerate <- NULL
  for (i in (1:(chartlength-1))){
    print(pricechart[i+1] / pricechart[i])
    pricechangerate <- c(pricechangerate, (pricechart[i+1] / pricechart[i]))
  }
  pricechangerate
}
{
  assign("x", 3)
}

simplify<- function(){
  for (stock in stocknames[functioniongnumber]){
    assign( paste("simple", stock, sep = ""), get(stock)[,4] )
  }
}

for (stock in stocknames[functioniongnumber]){
  assign( paste("simple", stock, sep = ""), get(stock)[,4] )
}
simplestocknames <- paste("simple", stocknames, sep = "")
distribute <- function(pricechart){
  len <- length(pricechart)
  increaserate <- NULL
  for (i in (1:(len -1))){
    increaserate <- c(increaserate, pricechart[i+1]/ pricechart[i] - 1)
  }
  resultmatrix <- matrix(data = 0, nrow = length(increaserate)-34, ncol = 35)
  for (i in 1:35){
    for (j  in 1:(length(increaserate)- 34)){
      resultmatrix[j,i] <- increaserate[i + j -1]
    }
  }
  resultmatrix
}

finaldata <- rbind(distribute( get(simplestocknames)))
finaldata <- read.table("Desktop/finaldata.txt")
write.table(finaldata,"Desktop/bitcoindata.txt", sep = " ", row.names = FALSE, col.names = FALSE)
write.table()
test<- rbind(distribute(simpleAABA), distribute(simpleAAL))

minedtext <- readLines("Desktop/python/news.txt")
refinedtext <- gsub(x = minedtext, pattern = "[^[:alnum:]]{1,}", " ")
refinedtext <- gsub(x = refinedtext, pattern = " {1,}", " ")
refinedtext <- refinedtext[refinedtext != " "]
refinedtext


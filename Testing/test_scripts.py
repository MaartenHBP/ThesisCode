from plots import IRAPlot

test = IRAPlot("test")

test.addPoint("algo 1", 1, 3)
test.addPoint("algo 1", 2, 10)
test.addPoint("algo 1", 3, 13)
test.addPoint("algo 1", 4, 36)
test.addPoint("algo 1", 5, 39)
test.addPoint("algo 1", 6, 60)
test.addPoint("algo 1", 7, 40)

test.addPoint("algo 2", 1, 33)
test.addPoint("algo 2", 2, 40)
test.addPoint("algo 2", 3, 50)
test.addPoint("algo 2", 4, 51)
test.addPoint("algo 2", 5, 51)
test.addPoint("algo 2", 6, 8)
test.addPoint("algo 2", 7, 13)

test.viewPlot()

test1 = IRAPlot("test")

test1.addPoint("algo 1", 1, 3)
test1.addPoint("algo 1", 2, 10)
test1.addPoint("algo 1", 3, 13)
test1.addPoint("algo 1", 4, 36)
test1.addPoint("algo 1", 5, 39)
test1.addPoint("algo 1", 6, 60)
test1.addPoint("algo 1", 7, 40)

test1.viewPlot()

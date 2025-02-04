# Visit 3.3.1 log file
ScriptVersion = "3.3.1"
if ScriptVersion != Version():
    print "This script is for VisIt %s. It may not work with version %s" % (ScriptVersion, Version())
ShowAllWindows()
AnnotationAtts = AnnotationAttributes()
AnnotationAtts.axes2D.visible = 1
AnnotationAtts.axes2D.autoSetTicks = 1
AnnotationAtts.axes2D.autoSetScaling = 1
AnnotationAtts.axes2D.lineWidth = 0
AnnotationAtts.axes2D.tickLocation = AnnotationAtts.axes2D.Outside  # Inside, Outside, Both
AnnotationAtts.axes2D.tickAxes = AnnotationAtts.axes2D.BottomLeft  # Off, Bottom, Left, BottomLeft, All
AnnotationAtts.axes2D.xAxis.title.visible = 1
AnnotationAtts.axes2D.xAxis.title.font.font = AnnotationAtts.axes2D.xAxis.title.font.Courier  # Arial, Courier, Times
AnnotationAtts.axes2D.xAxis.title.font.scale = 1
AnnotationAtts.axes2D.xAxis.title.font.useForegroundColor = 1
AnnotationAtts.axes2D.xAxis.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axes2D.xAxis.title.font.bold = 1
AnnotationAtts.axes2D.xAxis.title.font.italic = 1
AnnotationAtts.axes2D.xAxis.title.userTitle = 0
AnnotationAtts.axes2D.xAxis.title.userUnits = 0
AnnotationAtts.axes2D.xAxis.title.title = "X-Axis"
AnnotationAtts.axes2D.xAxis.title.units = ""
AnnotationAtts.axes2D.xAxis.label.visible = 1
AnnotationAtts.axes2D.xAxis.label.font.font = AnnotationAtts.axes2D.xAxis.label.font.Courier  # Arial, Courier, Times
AnnotationAtts.axes2D.xAxis.label.font.scale = 1
AnnotationAtts.axes2D.xAxis.label.font.useForegroundColor = 1
AnnotationAtts.axes2D.xAxis.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axes2D.xAxis.label.font.bold = 1
AnnotationAtts.axes2D.xAxis.label.font.italic = 1
AnnotationAtts.axes2D.xAxis.label.scaling = 0
AnnotationAtts.axes2D.xAxis.tickMarks.visible = 1
AnnotationAtts.axes2D.xAxis.tickMarks.majorMinimum = 0
AnnotationAtts.axes2D.xAxis.tickMarks.majorMaximum = 1
AnnotationAtts.axes2D.xAxis.tickMarks.minorSpacing = 0.02
AnnotationAtts.axes2D.xAxis.tickMarks.majorSpacing = 0.2
AnnotationAtts.axes2D.xAxis.grid = 0
AnnotationAtts.axes2D.yAxis.title.visible = 1
AnnotationAtts.axes2D.yAxis.title.font.font = AnnotationAtts.axes2D.yAxis.title.font.Courier  # Arial, Courier, Times
AnnotationAtts.axes2D.yAxis.title.font.scale = 1
AnnotationAtts.axes2D.yAxis.title.font.useForegroundColor = 1
AnnotationAtts.axes2D.yAxis.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axes2D.yAxis.title.font.bold = 1
AnnotationAtts.axes2D.yAxis.title.font.italic = 1
AnnotationAtts.axes2D.yAxis.title.userTitle = 0
AnnotationAtts.axes2D.yAxis.title.userUnits = 0
AnnotationAtts.axes2D.yAxis.title.title = "Y-Axis"
AnnotationAtts.axes2D.yAxis.title.units = ""
AnnotationAtts.axes2D.yAxis.label.visible = 1
AnnotationAtts.axes2D.yAxis.label.font.font = AnnotationAtts.axes2D.yAxis.label.font.Courier  # Arial, Courier, Times
AnnotationAtts.axes2D.yAxis.label.font.scale = 1
AnnotationAtts.axes2D.yAxis.label.font.useForegroundColor = 1
AnnotationAtts.axes2D.yAxis.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axes2D.yAxis.label.font.bold = 1
AnnotationAtts.axes2D.yAxis.label.font.italic = 1
AnnotationAtts.axes2D.yAxis.label.scaling = 0
AnnotationAtts.axes2D.yAxis.tickMarks.visible = 1
AnnotationAtts.axes2D.yAxis.tickMarks.majorMinimum = 0
AnnotationAtts.axes2D.yAxis.tickMarks.majorMaximum = 1
AnnotationAtts.axes2D.yAxis.tickMarks.minorSpacing = 0.02
AnnotationAtts.axes2D.yAxis.tickMarks.majorSpacing = 0.2
AnnotationAtts.axes2D.yAxis.grid = 0
AnnotationAtts.axes3D.visible = 0
AnnotationAtts.axes3D.autoSetTicks = 0
AnnotationAtts.axes3D.autoSetScaling = 0
AnnotationAtts.axes3D.lineWidth = 0
AnnotationAtts.axes3D.tickLocation = AnnotationAtts.axes3D.Inside  # Inside, Outside, Both
AnnotationAtts.axes3D.axesType = AnnotationAtts.axes3D.ClosestTriad  # ClosestTriad, FurthestTriad, OutsideEdges, StaticTriad, StaticEdges
AnnotationAtts.axes3D.triadFlag = 1
AnnotationAtts.axes3D.bboxFlag = 0
AnnotationAtts.axes3D.xAxis.title.visible = 1
AnnotationAtts.axes3D.xAxis.title.font.font = AnnotationAtts.axes3D.xAxis.title.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.xAxis.title.font.scale = 1
AnnotationAtts.axes3D.xAxis.title.font.useForegroundColor = 1
AnnotationAtts.axes3D.xAxis.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.xAxis.title.font.bold = 0
AnnotationAtts.axes3D.xAxis.title.font.italic = 0
AnnotationAtts.axes3D.xAxis.title.userTitle = 0
AnnotationAtts.axes3D.xAxis.title.userUnits = 0
AnnotationAtts.axes3D.xAxis.title.title = "X-Axis"
AnnotationAtts.axes3D.xAxis.title.units = ""
AnnotationAtts.axes3D.xAxis.label.visible = 1
AnnotationAtts.axes3D.xAxis.label.font.font = AnnotationAtts.axes3D.xAxis.label.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.xAxis.label.font.scale = 1
AnnotationAtts.axes3D.xAxis.label.font.useForegroundColor = 1
AnnotationAtts.axes3D.xAxis.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.xAxis.label.font.bold = 0
AnnotationAtts.axes3D.xAxis.label.font.italic = 0
AnnotationAtts.axes3D.xAxis.label.scaling = 0
AnnotationAtts.axes3D.xAxis.tickMarks.visible = 1
AnnotationAtts.axes3D.xAxis.tickMarks.majorMinimum = 0
AnnotationAtts.axes3D.xAxis.tickMarks.majorMaximum = 1
AnnotationAtts.axes3D.xAxis.tickMarks.minorSpacing = 0.02
AnnotationAtts.axes3D.xAxis.tickMarks.majorSpacing = 0.2
AnnotationAtts.axes3D.xAxis.grid = 0
AnnotationAtts.axes3D.yAxis.title.visible = 1
AnnotationAtts.axes3D.yAxis.title.font.font = AnnotationAtts.axes3D.yAxis.title.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.yAxis.title.font.scale = 1
AnnotationAtts.axes3D.yAxis.title.font.useForegroundColor = 1
AnnotationAtts.axes3D.yAxis.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.yAxis.title.font.bold = 0
AnnotationAtts.axes3D.yAxis.title.font.italic = 0
AnnotationAtts.axes3D.yAxis.title.userTitle = 0
AnnotationAtts.axes3D.yAxis.title.userUnits = 0
AnnotationAtts.axes3D.yAxis.title.title = "Y-Axis"
AnnotationAtts.axes3D.yAxis.title.units = ""
AnnotationAtts.axes3D.yAxis.label.visible = 1
AnnotationAtts.axes3D.yAxis.label.font.font = AnnotationAtts.axes3D.yAxis.label.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.yAxis.label.font.scale = 1
AnnotationAtts.axes3D.yAxis.label.font.useForegroundColor = 1
AnnotationAtts.axes3D.yAxis.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.yAxis.label.font.bold = 0
AnnotationAtts.axes3D.yAxis.label.font.italic = 0
AnnotationAtts.axes3D.yAxis.label.scaling = 0
AnnotationAtts.axes3D.yAxis.tickMarks.visible = 1
AnnotationAtts.axes3D.yAxis.tickMarks.majorMinimum = 0
AnnotationAtts.axes3D.yAxis.tickMarks.majorMaximum = 1
AnnotationAtts.axes3D.yAxis.tickMarks.minorSpacing = 0.02
AnnotationAtts.axes3D.yAxis.tickMarks.majorSpacing = 0.2
AnnotationAtts.axes3D.yAxis.grid = 0
AnnotationAtts.axes3D.zAxis.title.visible = 1
AnnotationAtts.axes3D.zAxis.title.font.font = AnnotationAtts.axes3D.zAxis.title.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.zAxis.title.font.scale = 1
AnnotationAtts.axes3D.zAxis.title.font.useForegroundColor = 1
AnnotationAtts.axes3D.zAxis.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.zAxis.title.font.bold = 0
AnnotationAtts.axes3D.zAxis.title.font.italic = 0
AnnotationAtts.axes3D.zAxis.title.userTitle = 0
AnnotationAtts.axes3D.zAxis.title.userUnits = 0
AnnotationAtts.axes3D.zAxis.title.title = "Z-Axis"
AnnotationAtts.axes3D.zAxis.title.units = ""
AnnotationAtts.axes3D.zAxis.label.visible = 1
AnnotationAtts.axes3D.zAxis.label.font.font = AnnotationAtts.axes3D.zAxis.label.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.zAxis.label.font.scale = 1
AnnotationAtts.axes3D.zAxis.label.font.useForegroundColor = 1
AnnotationAtts.axes3D.zAxis.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.zAxis.label.font.bold = 0
AnnotationAtts.axes3D.zAxis.label.font.italic = 0
AnnotationAtts.axes3D.zAxis.label.scaling = 1
AnnotationAtts.axes3D.zAxis.tickMarks.visible = 1
AnnotationAtts.axes3D.zAxis.tickMarks.majorMinimum = 0
AnnotationAtts.axes3D.zAxis.tickMarks.majorMaximum = 1
AnnotationAtts.axes3D.zAxis.tickMarks.minorSpacing = 0.02
AnnotationAtts.axes3D.zAxis.tickMarks.majorSpacing = 0.2
AnnotationAtts.axes3D.zAxis.grid = 0
AnnotationAtts.axes3D.setBBoxLocation = 0
AnnotationAtts.axes3D.bboxLocation = (0, 1, 0, 1, 0, 1)
AnnotationAtts.axes3D.triadColor = (0, 0, 0)
AnnotationAtts.axes3D.triadLineWidth = 0
AnnotationAtts.axes3D.triadFont = 0
AnnotationAtts.axes3D.triadBold = 1
AnnotationAtts.axes3D.triadItalic = 1
AnnotationAtts.axes3D.triadSetManually = 0
AnnotationAtts.userInfoFlag = 1
AnnotationAtts.userInfoFont.font = AnnotationAtts.userInfoFont.Courier  # Arial, Courier, Times
AnnotationAtts.userInfoFont.scale = 1
AnnotationAtts.userInfoFont.useForegroundColor = 1
AnnotationAtts.userInfoFont.color = (0, 0, 0, 255)
AnnotationAtts.userInfoFont.bold = 0
AnnotationAtts.userInfoFont.italic = 0
AnnotationAtts.databaseInfoFlag = 1
AnnotationAtts.timeInfoFlag = 1
AnnotationAtts.databaseInfoFont.font = AnnotationAtts.databaseInfoFont.Courier  # Arial, Courier, Times
AnnotationAtts.databaseInfoFont.scale = 1
AnnotationAtts.databaseInfoFont.useForegroundColor = 1
AnnotationAtts.databaseInfoFont.color = (0, 0, 0, 255)
AnnotationAtts.databaseInfoFont.bold = 0
AnnotationAtts.databaseInfoFont.italic = 0
AnnotationAtts.databaseInfoExpansionMode = AnnotationAtts.File  # File, Directory, Full, Smart, SmartDirectory
AnnotationAtts.databaseInfoTimeScale = 1
AnnotationAtts.databaseInfoTimeOffset = 0
AnnotationAtts.legendInfoFlag = 1
AnnotationAtts.backgroundColor = (255, 255, 255, 255)
AnnotationAtts.foregroundColor = (153, 204, 255, 255)
AnnotationAtts.gradientBackgroundStyle = AnnotationAtts.TopToBottom  # TopToBottom, BottomToTop, LeftToRight, RightToLeft, Radial
AnnotationAtts.gradientColor1 = (0, 0, 0, 255)
AnnotationAtts.gradientColor2 = (0, 51, 102, 255)
AnnotationAtts.backgroundMode = AnnotationAtts.Gradient  # Solid, Gradient, Image, ImageSphere
AnnotationAtts.backgroundImage = ""
AnnotationAtts.imageRepeatX = 1
AnnotationAtts.imageRepeatY = 1
AnnotationAtts.axesArray.visible = 1
AnnotationAtts.axesArray.ticksVisible = 1
AnnotationAtts.axesArray.autoSetTicks = 1
AnnotationAtts.axesArray.autoSetScaling = 1
AnnotationAtts.axesArray.lineWidth = 0
AnnotationAtts.axesArray.axes.title.visible = 1
AnnotationAtts.axesArray.axes.title.font.font = AnnotationAtts.axesArray.axes.title.font.Arial  # Arial, Courier, Times
AnnotationAtts.axesArray.axes.title.font.scale = 1
AnnotationAtts.axesArray.axes.title.font.useForegroundColor = 1
AnnotationAtts.axesArray.axes.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axesArray.axes.title.font.bold = 0
AnnotationAtts.axesArray.axes.title.font.italic = 0
AnnotationAtts.axesArray.axes.title.userTitle = 0
AnnotationAtts.axesArray.axes.title.userUnits = 0
AnnotationAtts.axesArray.axes.title.title = ""
AnnotationAtts.axesArray.axes.title.units = ""
AnnotationAtts.axesArray.axes.label.visible = 1
AnnotationAtts.axesArray.axes.label.font.font = AnnotationAtts.axesArray.axes.label.font.Arial  # Arial, Courier, Times
AnnotationAtts.axesArray.axes.label.font.scale = 1
AnnotationAtts.axesArray.axes.label.font.useForegroundColor = 1
AnnotationAtts.axesArray.axes.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axesArray.axes.label.font.bold = 0
AnnotationAtts.axesArray.axes.label.font.italic = 0
AnnotationAtts.axesArray.axes.label.scaling = 0
AnnotationAtts.axesArray.axes.tickMarks.visible = 1
AnnotationAtts.axesArray.axes.tickMarks.majorMinimum = 0
AnnotationAtts.axesArray.axes.tickMarks.majorMaximum = 1
AnnotationAtts.axesArray.axes.tickMarks.minorSpacing = 0.02
AnnotationAtts.axesArray.axes.tickMarks.majorSpacing = 0.2
AnnotationAtts.axesArray.axes.grid = 0
SetAnnotationAttributes(AnnotationAtts)
AnnotationAtts = AnnotationAttributes()
AnnotationAtts.axes2D.visible = 1
AnnotationAtts.axes2D.autoSetTicks = 1
AnnotationAtts.axes2D.autoSetScaling = 1
AnnotationAtts.axes2D.lineWidth = 0
AnnotationAtts.axes2D.tickLocation = AnnotationAtts.axes2D.Outside  # Inside, Outside, Both
AnnotationAtts.axes2D.tickAxes = AnnotationAtts.axes2D.BottomLeft  # Off, Bottom, Left, BottomLeft, All
AnnotationAtts.axes2D.xAxis.title.visible = 1
AnnotationAtts.axes2D.xAxis.title.font.font = AnnotationAtts.axes2D.xAxis.title.font.Courier  # Arial, Courier, Times
AnnotationAtts.axes2D.xAxis.title.font.scale = 1
AnnotationAtts.axes2D.xAxis.title.font.useForegroundColor = 1
AnnotationAtts.axes2D.xAxis.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axes2D.xAxis.title.font.bold = 1
AnnotationAtts.axes2D.xAxis.title.font.italic = 1
AnnotationAtts.axes2D.xAxis.title.userTitle = 0
AnnotationAtts.axes2D.xAxis.title.userUnits = 0
AnnotationAtts.axes2D.xAxis.title.title = "X-Axis"
AnnotationAtts.axes2D.xAxis.title.units = ""
AnnotationAtts.axes2D.xAxis.label.visible = 1
AnnotationAtts.axes2D.xAxis.label.font.font = AnnotationAtts.axes2D.xAxis.label.font.Courier  # Arial, Courier, Times
AnnotationAtts.axes2D.xAxis.label.font.scale = 1
AnnotationAtts.axes2D.xAxis.label.font.useForegroundColor = 1
AnnotationAtts.axes2D.xAxis.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axes2D.xAxis.label.font.bold = 1
AnnotationAtts.axes2D.xAxis.label.font.italic = 1
AnnotationAtts.axes2D.xAxis.label.scaling = 0
AnnotationAtts.axes2D.xAxis.tickMarks.visible = 1
AnnotationAtts.axes2D.xAxis.tickMarks.majorMinimum = 0
AnnotationAtts.axes2D.xAxis.tickMarks.majorMaximum = 1
AnnotationAtts.axes2D.xAxis.tickMarks.minorSpacing = 0.02
AnnotationAtts.axes2D.xAxis.tickMarks.majorSpacing = 0.2
AnnotationAtts.axes2D.xAxis.grid = 0
AnnotationAtts.axes2D.yAxis.title.visible = 1
AnnotationAtts.axes2D.yAxis.title.font.font = AnnotationAtts.axes2D.yAxis.title.font.Courier  # Arial, Courier, Times
AnnotationAtts.axes2D.yAxis.title.font.scale = 1
AnnotationAtts.axes2D.yAxis.title.font.useForegroundColor = 1
AnnotationAtts.axes2D.yAxis.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axes2D.yAxis.title.font.bold = 1
AnnotationAtts.axes2D.yAxis.title.font.italic = 1
AnnotationAtts.axes2D.yAxis.title.userTitle = 0
AnnotationAtts.axes2D.yAxis.title.userUnits = 0
AnnotationAtts.axes2D.yAxis.title.title = "Y-Axis"
AnnotationAtts.axes2D.yAxis.title.units = ""
AnnotationAtts.axes2D.yAxis.label.visible = 1
AnnotationAtts.axes2D.yAxis.label.font.font = AnnotationAtts.axes2D.yAxis.label.font.Courier  # Arial, Courier, Times
AnnotationAtts.axes2D.yAxis.label.font.scale = 1
AnnotationAtts.axes2D.yAxis.label.font.useForegroundColor = 1
AnnotationAtts.axes2D.yAxis.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axes2D.yAxis.label.font.bold = 1
AnnotationAtts.axes2D.yAxis.label.font.italic = 1
AnnotationAtts.axes2D.yAxis.label.scaling = 0
AnnotationAtts.axes2D.yAxis.tickMarks.visible = 1
AnnotationAtts.axes2D.yAxis.tickMarks.majorMinimum = 0
AnnotationAtts.axes2D.yAxis.tickMarks.majorMaximum = 1
AnnotationAtts.axes2D.yAxis.tickMarks.minorSpacing = 0.02
AnnotationAtts.axes2D.yAxis.tickMarks.majorSpacing = 0.2
AnnotationAtts.axes2D.yAxis.grid = 0
AnnotationAtts.axes3D.visible = 0
AnnotationAtts.axes3D.autoSetTicks = 0
AnnotationAtts.axes3D.autoSetScaling = 0
AnnotationAtts.axes3D.lineWidth = 0
AnnotationAtts.axes3D.tickLocation = AnnotationAtts.axes3D.Inside  # Inside, Outside, Both
AnnotationAtts.axes3D.axesType = AnnotationAtts.axes3D.ClosestTriad  # ClosestTriad, FurthestTriad, OutsideEdges, StaticTriad, StaticEdges
AnnotationAtts.axes3D.triadFlag = 1
AnnotationAtts.axes3D.bboxFlag = 0
AnnotationAtts.axes3D.xAxis.title.visible = 1
AnnotationAtts.axes3D.xAxis.title.font.font = AnnotationAtts.axes3D.xAxis.title.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.xAxis.title.font.scale = 1
AnnotationAtts.axes3D.xAxis.title.font.useForegroundColor = 1
AnnotationAtts.axes3D.xAxis.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.xAxis.title.font.bold = 0
AnnotationAtts.axes3D.xAxis.title.font.italic = 0
AnnotationAtts.axes3D.xAxis.title.userTitle = 0
AnnotationAtts.axes3D.xAxis.title.userUnits = 0
AnnotationAtts.axes3D.xAxis.title.title = "X-Axis"
AnnotationAtts.axes3D.xAxis.title.units = ""
AnnotationAtts.axes3D.xAxis.label.visible = 1
AnnotationAtts.axes3D.xAxis.label.font.font = AnnotationAtts.axes3D.xAxis.label.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.xAxis.label.font.scale = 1
AnnotationAtts.axes3D.xAxis.label.font.useForegroundColor = 1
AnnotationAtts.axes3D.xAxis.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.xAxis.label.font.bold = 0
AnnotationAtts.axes3D.xAxis.label.font.italic = 0
AnnotationAtts.axes3D.xAxis.label.scaling = 0
AnnotationAtts.axes3D.xAxis.tickMarks.visible = 1
AnnotationAtts.axes3D.xAxis.tickMarks.majorMinimum = 0
AnnotationAtts.axes3D.xAxis.tickMarks.majorMaximum = 1
AnnotationAtts.axes3D.xAxis.tickMarks.minorSpacing = 0.02
AnnotationAtts.axes3D.xAxis.tickMarks.majorSpacing = 0.2
AnnotationAtts.axes3D.xAxis.grid = 0
AnnotationAtts.axes3D.yAxis.title.visible = 1
AnnotationAtts.axes3D.yAxis.title.font.font = AnnotationAtts.axes3D.yAxis.title.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.yAxis.title.font.scale = 1
AnnotationAtts.axes3D.yAxis.title.font.useForegroundColor = 1
AnnotationAtts.axes3D.yAxis.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.yAxis.title.font.bold = 0
AnnotationAtts.axes3D.yAxis.title.font.italic = 0
AnnotationAtts.axes3D.yAxis.title.userTitle = 0
AnnotationAtts.axes3D.yAxis.title.userUnits = 0
AnnotationAtts.axes3D.yAxis.title.title = "Y-Axis"
AnnotationAtts.axes3D.yAxis.title.units = ""
AnnotationAtts.axes3D.yAxis.label.visible = 1
AnnotationAtts.axes3D.yAxis.label.font.font = AnnotationAtts.axes3D.yAxis.label.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.yAxis.label.font.scale = 1
AnnotationAtts.axes3D.yAxis.label.font.useForegroundColor = 1
AnnotationAtts.axes3D.yAxis.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.yAxis.label.font.bold = 0
AnnotationAtts.axes3D.yAxis.label.font.italic = 0
AnnotationAtts.axes3D.yAxis.label.scaling = 0
AnnotationAtts.axes3D.yAxis.tickMarks.visible = 1
AnnotationAtts.axes3D.yAxis.tickMarks.majorMinimum = 0
AnnotationAtts.axes3D.yAxis.tickMarks.majorMaximum = 1
AnnotationAtts.axes3D.yAxis.tickMarks.minorSpacing = 0.02
AnnotationAtts.axes3D.yAxis.tickMarks.majorSpacing = 0.2
AnnotationAtts.axes3D.yAxis.grid = 0
AnnotationAtts.axes3D.zAxis.title.visible = 1
AnnotationAtts.axes3D.zAxis.title.font.font = AnnotationAtts.axes3D.zAxis.title.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.zAxis.title.font.scale = 1
AnnotationAtts.axes3D.zAxis.title.font.useForegroundColor = 1
AnnotationAtts.axes3D.zAxis.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.zAxis.title.font.bold = 0
AnnotationAtts.axes3D.zAxis.title.font.italic = 0
AnnotationAtts.axes3D.zAxis.title.userTitle = 0
AnnotationAtts.axes3D.zAxis.title.userUnits = 0
AnnotationAtts.axes3D.zAxis.title.title = "Z-Axis"
AnnotationAtts.axes3D.zAxis.title.units = ""
AnnotationAtts.axes3D.zAxis.label.visible = 1
AnnotationAtts.axes3D.zAxis.label.font.font = AnnotationAtts.axes3D.zAxis.label.font.Arial  # Arial, Courier, Times
AnnotationAtts.axes3D.zAxis.label.font.scale = 1
AnnotationAtts.axes3D.zAxis.label.font.useForegroundColor = 1
AnnotationAtts.axes3D.zAxis.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axes3D.zAxis.label.font.bold = 0
AnnotationAtts.axes3D.zAxis.label.font.italic = 0
AnnotationAtts.axes3D.zAxis.label.scaling = 1
AnnotationAtts.axes3D.zAxis.tickMarks.visible = 1
AnnotationAtts.axes3D.zAxis.tickMarks.majorMinimum = 0
AnnotationAtts.axes3D.zAxis.tickMarks.majorMaximum = 1
AnnotationAtts.axes3D.zAxis.tickMarks.minorSpacing = 0.02
AnnotationAtts.axes3D.zAxis.tickMarks.majorSpacing = 0.2
AnnotationAtts.axes3D.zAxis.grid = 0
AnnotationAtts.axes3D.setBBoxLocation = 0
AnnotationAtts.axes3D.bboxLocation = (0, 1, 0, 1, 0, 1)
AnnotationAtts.axes3D.triadColor = (0, 0, 0)
AnnotationAtts.axes3D.triadLineWidth = 0
AnnotationAtts.axes3D.triadFont = 0
AnnotationAtts.axes3D.triadBold = 1
AnnotationAtts.axes3D.triadItalic = 1
AnnotationAtts.axes3D.triadSetManually = 0
AnnotationAtts.userInfoFlag = 1
AnnotationAtts.userInfoFont.font = AnnotationAtts.userInfoFont.Courier  # Arial, Courier, Times
AnnotationAtts.userInfoFont.scale = 1
AnnotationAtts.userInfoFont.useForegroundColor = 1
AnnotationAtts.userInfoFont.color = (0, 0, 0, 255)
AnnotationAtts.userInfoFont.bold = 0
AnnotationAtts.userInfoFont.italic = 0
AnnotationAtts.databaseInfoFlag = 1
AnnotationAtts.timeInfoFlag = 1
AnnotationAtts.databaseInfoFont.font = AnnotationAtts.databaseInfoFont.Courier  # Arial, Courier, Times
AnnotationAtts.databaseInfoFont.scale = 1
AnnotationAtts.databaseInfoFont.useForegroundColor = 1
AnnotationAtts.databaseInfoFont.color = (0, 0, 0, 255)
AnnotationAtts.databaseInfoFont.bold = 0
AnnotationAtts.databaseInfoFont.italic = 0
AnnotationAtts.databaseInfoExpansionMode = AnnotationAtts.File  # File, Directory, Full, Smart, SmartDirectory
AnnotationAtts.databaseInfoTimeScale = 1
AnnotationAtts.databaseInfoTimeOffset = 0
AnnotationAtts.legendInfoFlag = 1
AnnotationAtts.backgroundColor = (255, 255, 255, 255)
AnnotationAtts.foregroundColor = (153, 204, 255, 255)
AnnotationAtts.gradientBackgroundStyle = AnnotationAtts.TopToBottom  # TopToBottom, BottomToTop, LeftToRight, RightToLeft, Radial
AnnotationAtts.gradientColor1 = (0, 0, 0, 255)
AnnotationAtts.gradientColor2 = (0, 51, 102, 255)
AnnotationAtts.backgroundMode = AnnotationAtts.Gradient  # Solid, Gradient, Image, ImageSphere
AnnotationAtts.backgroundImage = ""
AnnotationAtts.imageRepeatX = 1
AnnotationAtts.imageRepeatY = 1
AnnotationAtts.axesArray.visible = 1
AnnotationAtts.axesArray.ticksVisible = 1
AnnotationAtts.axesArray.autoSetTicks = 1
AnnotationAtts.axesArray.autoSetScaling = 1
AnnotationAtts.axesArray.lineWidth = 0
AnnotationAtts.axesArray.axes.title.visible = 1
AnnotationAtts.axesArray.axes.title.font.font = AnnotationAtts.axesArray.axes.title.font.Arial  # Arial, Courier, Times
AnnotationAtts.axesArray.axes.title.font.scale = 1
AnnotationAtts.axesArray.axes.title.font.useForegroundColor = 1
AnnotationAtts.axesArray.axes.title.font.color = (0, 0, 0, 255)
AnnotationAtts.axesArray.axes.title.font.bold = 0
AnnotationAtts.axesArray.axes.title.font.italic = 0
AnnotationAtts.axesArray.axes.title.userTitle = 0
AnnotationAtts.axesArray.axes.title.userUnits = 0
AnnotationAtts.axesArray.axes.title.title = ""
AnnotationAtts.axesArray.axes.title.units = ""
AnnotationAtts.axesArray.axes.label.visible = 1
AnnotationAtts.axesArray.axes.label.font.font = AnnotationAtts.axesArray.axes.label.font.Arial  # Arial, Courier, Times
AnnotationAtts.axesArray.axes.label.font.scale = 1
AnnotationAtts.axesArray.axes.label.font.useForegroundColor = 1
AnnotationAtts.axesArray.axes.label.font.color = (0, 0, 0, 255)
AnnotationAtts.axesArray.axes.label.font.bold = 0
AnnotationAtts.axesArray.axes.label.font.italic = 0
AnnotationAtts.axesArray.axes.label.scaling = 0
AnnotationAtts.axesArray.axes.tickMarks.visible = 1
AnnotationAtts.axesArray.axes.tickMarks.majorMinimum = 0
AnnotationAtts.axesArray.axes.tickMarks.majorMaximum = 1
AnnotationAtts.axesArray.axes.tickMarks.minorSpacing = 0.02
AnnotationAtts.axesArray.axes.tickMarks.majorSpacing = 0.2
AnnotationAtts.axesArray.axes.grid = 0
SetDefaultAnnotationAttributes(AnnotationAtts)
AddOperator("Displace", 1)
RemoveLastOperator(1)
DrawPlots()
SetActivePlots(0)
SetActivePlots(0)
AddPlot("Pseudocolor", "x", 1, 1)
DrawPlots()
# Begin spontaneous state
View3DAtts = View3DAttributes()
View3DAtts.viewNormal = (0.479205, 0.147043, 0.865298)
View3DAtts.focus = (50, 50, 250)
View3DAtts.viewUp = (0.0931876, 0.971771, -0.216744)
View3DAtts.viewAngle = 30
View3DAtts.parallelScale = 259.808
View3DAtts.nearPlane = -519.615
View3DAtts.farPlane = 519.615
View3DAtts.imagePan = (0, 0)
View3DAtts.imageZoom = 1
View3DAtts.perspective = 1
View3DAtts.eyeAngle = 2
View3DAtts.centerOfRotationSet = 0
View3DAtts.centerOfRotation = (50, 50, 250)
View3DAtts.axis3DScaleFlag = 0
View3DAtts.axis3DScales = (1, 1, 1)
View3DAtts.shear = (0, 0, 1)
View3DAtts.windowValid = 1
SetView3D(View3DAtts)
# End spontaneous state

AddOperator("Displace", 1)
SetActivePlots((0, 1))
SetActivePlots(0)
SetActivePlots(0)
SetActivePlots(0)
SetActivePlots((0, 1))
SetActivePlots(1)
ChangeActivePlotsVar("Z")
DrawPlots()
RemoveAllOperators(1)
DrawPlots()
# Begin spontaneous state
View3DAtts = View3DAttributes()
View3DAtts.viewNormal = (0.479205, 0.147043, 0.865298)
View3DAtts.focus = (50, 50, 250)
View3DAtts.viewUp = (0.0931876, 0.971771, -0.216744)
View3DAtts.viewAngle = 30
View3DAtts.parallelScale = 259.808
View3DAtts.nearPlane = -519.615
View3DAtts.farPlane = 519.615
View3DAtts.imagePan = (0, 0)
View3DAtts.imageZoom = 1.21
View3DAtts.perspective = 1
View3DAtts.eyeAngle = 2
View3DAtts.centerOfRotationSet = 0
View3DAtts.centerOfRotation = (50, 50, 250)
View3DAtts.axis3DScaleFlag = 0
View3DAtts.axis3DScales = (1, 1, 1)
View3DAtts.shear = (0, 0, 1)
View3DAtts.windowValid = 1
SetView3D(View3DAtts)
# End spontaneous state

# Begin spontaneous state
View3DAtts = View3DAttributes()
View3DAtts.viewNormal = (0.479205, 0.147043, 0.865298)
View3DAtts.focus = (50, 50, 250)
View3DAtts.viewUp = (0.0931876, 0.971771, -0.216744)
View3DAtts.viewAngle = 30
View3DAtts.parallelScale = 259.808
View3DAtts.nearPlane = -519.615
View3DAtts.farPlane = 519.615
View3DAtts.imagePan = (0, 0)
View3DAtts.imageZoom = 1
View3DAtts.perspective = 1
View3DAtts.eyeAngle = 2
View3DAtts.centerOfRotationSet = 0
View3DAtts.centerOfRotation = (50, 50, 250)
View3DAtts.axis3DScaleFlag = 0
View3DAtts.axis3DScales = (1, 1, 1)
View3DAtts.shear = (0, 0, 1)
View3DAtts.windowValid = 1
SetView3D(View3DAtts)
# End spontaneous state

# Begin spontaneous state
View3DAtts = View3DAttributes()
View3DAtts.viewNormal = (0.479205, 0.147043, 0.865298)
View3DAtts.focus = (50, 50, 250)
View3DAtts.viewUp = (0.0931876, 0.971771, -0.216744)
View3DAtts.viewAngle = 30
View3DAtts.parallelScale = 259.808
View3DAtts.nearPlane = -519.615
View3DAtts.farPlane = 519.615
View3DAtts.imagePan = (0, 0)
View3DAtts.imageZoom = 0.826446
View3DAtts.perspective = 1
View3DAtts.eyeAngle = 2
View3DAtts.centerOfRotationSet = 0
View3DAtts.centerOfRotation = (50, 50, 250)
View3DAtts.axis3DScaleFlag = 0
View3DAtts.axis3DScales = (1, 1, 1)
View3DAtts.shear = (0, 0, 1)
View3DAtts.windowValid = 1
SetView3D(View3DAtts)
# End spontaneous state

# Begin spontaneous state
View3DAtts = View3DAttributes()
View3DAtts.viewNormal = (-0.603976, 0.608568, 0.514643)
View3DAtts.focus = (50, 50, 250)
View3DAtts.viewUp = (0.472799, 0.793414, -0.383347)
View3DAtts.viewAngle = 30
View3DAtts.parallelScale = 259.808
View3DAtts.nearPlane = -519.615
View3DAtts.farPlane = 519.615
View3DAtts.imagePan = (0, 0)
View3DAtts.imageZoom = 0.826446
View3DAtts.perspective = 1
View3DAtts.eyeAngle = 2
View3DAtts.centerOfRotationSet = 0
View3DAtts.centerOfRotation = (50, 50, 250)
View3DAtts.axis3DScaleFlag = 0
View3DAtts.axis3DScales = (1, 1, 1)
View3DAtts.shear = (0, 0, 1)
View3DAtts.windowValid = 1
SetView3D(View3DAtts)
# End spontaneous state

SetActivePlots(1)
SetActivePlots(1)
ChangeActivePlotsVar("Y")
ChangeActivePlotsVar("Z")
SetActivePlots((0, 1))
SetActivePlots(0)
AddOperator("Displace", 0)
DefineScalarExpression("operators/ConnectedComponents/mesh", "cell_constant(<mesh>, 0.)")
DefineCurveExpression("operators/DataBinning/1D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/DataBinning/2D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/DataBinning/3D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/Flux/mesh", "cell_constant(<mesh>, 0.)")
DefineCurveExpression("operators/Lineout/Y", "cell_constant(<Y>, 0.)")
DefineCurveExpression("operators/Lineout/Z", "cell_constant(<Z>, 0.)")
DefineCurveExpression("operators/Lineout/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/ModelFit/distance", "point_constant(<mesh>, 0)")
DefineScalarExpression("operators/ModelFit/model", "point_constant(<mesh>, 0)")
DefineScalarExpression("operators/StatisticalTrends/Mean/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Mean/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Mean/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/x", "cell_constant(<x>, 0.)")
DefineVectorExpression("operators/SurfaceNormal/mesh", "cell_constant(<mesh>, 0.)")
DefineVectorExpression("unnamed1", "Z")
DisplaceAtts = DisplaceAttributes()
DisplaceAtts.factor = 1
DisplaceAtts.variable = "unnamed1"
SetOperatorOptions(DisplaceAtts, 0, 0)
DrawPlots()
# MAINTENANCE ISSUE: SetSuppressMessagesRPC is not handled in Logging.C. Please contact a VisIt developer.
SaveSession("/home/honza/.visit/crash_recovery.116357.session")
# MAINTENANCE ISSUE: SetSuppressMessagesRPC is not handled in Logging.C. Please contact a VisIt developer.
DefineScalarExpression("operators/ConnectedComponents/mesh", "cell_constant(<mesh>, 0.)")
DefineCurveExpression("operators/DataBinning/1D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/DataBinning/2D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/DataBinning/3D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/Flux/mesh", "cell_constant(<mesh>, 0.)")
DefineCurveExpression("operators/Lineout/Y", "cell_constant(<Y>, 0.)")
DefineCurveExpression("operators/Lineout/Z", "cell_constant(<Z>, 0.)")
DefineCurveExpression("operators/Lineout/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/ModelFit/distance", "point_constant(<mesh>, 0)")
DefineScalarExpression("operators/ModelFit/model", "point_constant(<mesh>, 0)")
DefineScalarExpression("operators/StatisticalTrends/Mean/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Mean/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Mean/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/x", "cell_constant(<x>, 0.)")
DefineVectorExpression("operators/SurfaceNormal/mesh", "cell_constant(<mesh>, 0.)")
DefineVectorExpression("unnamed1", "array_compose(<x>, <Y>, .<Z>)")
DisplaceAtts = DisplaceAttributes()
DisplaceAtts.factor = 1
DisplaceAtts.variable = "unnamed1"
SetOperatorOptions(DisplaceAtts, 0, 0)
DisplaceAtts = DisplaceAttributes()
DisplaceAtts.factor = 1
DisplaceAtts.variable = "unnamed1"
SetOperatorOptions(DisplaceAtts, 0, 0)
DrawPlots()
DefineScalarExpression("operators/ConnectedComponents/mesh", "cell_constant(<mesh>, 0.)")
DefineCurveExpression("operators/DataBinning/1D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/DataBinning/2D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/DataBinning/3D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/Flux/mesh", "cell_constant(<mesh>, 0.)")
DefineCurveExpression("operators/Lineout/Y", "cell_constant(<Y>, 0.)")
DefineCurveExpression("operators/Lineout/Z", "cell_constant(<Z>, 0.)")
DefineCurveExpression("operators/Lineout/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/ModelFit/distance", "point_constant(<mesh>, 0)")
DefineScalarExpression("operators/ModelFit/model", "point_constant(<mesh>, 0)")
DefineScalarExpression("operators/StatisticalTrends/Mean/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Mean/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Mean/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/x", "cell_constant(<x>, 0.)")
DefineVectorExpression("operators/SurfaceNormal/mesh", "cell_constant(<mesh>, 0.)")
DefineVectorExpression("unnamed1", "array_compose(x, Y, Z)")
DefineScalarExpression("operators/ConnectedComponents/mesh", "cell_constant(<mesh>, 0.)")
DefineCurveExpression("operators/DataBinning/1D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/DataBinning/2D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/DataBinning/3D/mesh", "cell_constant(<mesh>, 0)")
DefineScalarExpression("operators/Flux/mesh", "cell_constant(<mesh>, 0.)")
DefineCurveExpression("operators/Lineout/Y", "cell_constant(<Y>, 0.)")
DefineCurveExpression("operators/Lineout/Z", "cell_constant(<Z>, 0.)")
DefineCurveExpression("operators/Lineout/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/ModelFit/distance", "point_constant(<mesh>, 0)")
DefineScalarExpression("operators/ModelFit/model", "point_constant(<mesh>, 0)")
DefineScalarExpression("operators/StatisticalTrends/Mean/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Mean/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Mean/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Residuals/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Slope/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Std. Dev./x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Sum/x", "cell_constant(<x>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/Y", "cell_constant(<Y>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/Z", "cell_constant(<Z>, 0.)")
DefineScalarExpression("operators/StatisticalTrends/Variance/x", "cell_constant(<x>, 0.)")
DefineVectorExpression("operators/SurfaceNormal/mesh", "cell_constant(<mesh>, 0.)")
DefineVectorExpression("unnamed1", "array_compose(x, Y, Z)")
DrawPlots()
# Begin spontaneous state
View3DAtts = View3DAttributes()
View3DAtts.viewNormal = (-0.603976, 0.608568, 0.514643)
View3DAtts.focus = (50, 50, 250)
View3DAtts.viewUp = (0.472799, 0.793414, -0.383347)
View3DAtts.viewAngle = 30
View3DAtts.parallelScale = 259.808
View3DAtts.nearPlane = -519.615
View3DAtts.farPlane = 519.615
View3DAtts.imagePan = (0, 0)
View3DAtts.imageZoom = 0.826446
View3DAtts.perspective = 1
View3DAtts.eyeAngle = 2
View3DAtts.centerOfRotationSet = 0
View3DAtts.centerOfRotation = (50, 50, 250)
View3DAtts.axis3DScaleFlag = 0
View3DAtts.axis3DScales = (1, 1, 1)
View3DAtts.shear = (0, 0, 1)
View3DAtts.windowValid = 1
SetView3D(View3DAtts)
# End spontaneous state

# Begin spontaneous state
View3DAtts = View3DAttributes()
View3DAtts.viewNormal = (-0.603976, 0.608568, 0.514643)
View3DAtts.focus = (50, 50, 250)
View3DAtts.viewUp = (0.472799, 0.793414, -0.383347)
View3DAtts.viewAngle = 30
View3DAtts.parallelScale = 259.808
View3DAtts.nearPlane = -519.615
View3DAtts.farPlane = 519.615
View3DAtts.imagePan = (0, 0)
View3DAtts.imageZoom = 0.826446
View3DAtts.perspective = 1
View3DAtts.eyeAngle = 2
View3DAtts.centerOfRotationSet = 0
View3DAtts.centerOfRotation = (50, 50, 250)
View3DAtts.axis3DScaleFlag = 0
View3DAtts.axis3DScales = (1, 1, 1)
View3DAtts.shear = (0, 0, 1)
View3DAtts.windowValid = 1
SetView3D(View3DAtts)
# End spontaneous state


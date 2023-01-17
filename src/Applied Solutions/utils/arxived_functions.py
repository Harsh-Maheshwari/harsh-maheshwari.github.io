def missing(df):
	# Capture the necessary data
	variables = df.columns

	count = []

	for variable in variables:
		length = df[variable].count()
		count.append(length)

	count_pct = np.round(100 * pd.Series(count) / len(df), 2)
	count = pd.Series(count)

	missing = pd.DataFrame()
	missing['variables'] = variables
	missing['count'] = len(df) - count
	missing['count_pct'] = 100 - count_pct
	missing = missing[missing['count_pct'] > 0]
	missing.sort_values(by=['count_pct'], inplace=True)

	# #Plot number of available data per variable
	# plt.subplots(figsize=(15,6))

	# # Plots missing data in percentage
	# plt.subplot(1,2,1)
	# plt.barh(missing['variables'], missing['count_pct'])
	# plt.title('Count of missing  data in percent', fontsize=15)

	# # Plots total row number of missing data
	# plt.subplot(1,2,2)
	# plt.barh(missing['variables'], missing['count'])
	# plt.title('Count of missing data as total records', fontsize=15)

	# plt.show(block=False)
	# plt.pause(3)
	# plt.close()
	miss = pd.DataFrame(list(missing['variables']),columns=['Features'])
	miss.insert(1, "Percentage", list(missing['count_pct']), True)
	miss.insert(2, "Count", list(missing['count']), True)
	print(miss)

	return miss    

def cleanup(test_file,target,char_var,df_train,df_test = None):

	# Missing Values
	print('Missing Data In the Train File')
	miss_train = missing(df_train)
	nan_var = list(miss_train[miss_train.Percentage>=40].Features)

	if test_file==True:
		print('Missing Data In the Test File')
		miss_test = missing(df_test)
		nan_var_test = list(miss_test[miss_test.Percentage>=40].Features)
		nan_var.extend(nan_var_test)
		del nan_var_test

	nan_var = set(nan_var)
	
	# Missing Value Treatement
	df_train = df_train.dropna(how = 'all') # Dropping rows if all values in that row are missing
	df_train = df_train.drop(nan_var,axis = 1) # removing more than 40 % nan Columns
	df_train = df_train.interpolate(method ='linear', limit_direction ='forward') # Rest nan values are interpolated

	# from sklearn.preprocessing import Imputer
	# imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

	df_train = df_train.fillna(method='pad')
	print('After Clenup Missing data in train')
	_ = missing(df_train)

	if test_file==True:
		df_test = df_test.dropna(how = 'all') # Dropping rows if all values in that row are missing
		df_test = df_test.drop(nan_var,axis = 1) # removing more than 50 % nan Columns
		df_test = df_test.interpolate(method ='linear', limit_direction ='forward') # Rest nan values are interpolated
		df_test = df_test.fillna(method='pad')
		print('After Clenup Missing data in test')
		_ = missing(df_test)


	# Correct Data Types in Categorical and Numerical Features
	variables = df_train.columns
	
	num_var = df_train.select_dtypes(include=['float64','float32','int32','int64']).columns
	try:
		num_var = num_var.drop(char_var)
		print("char_var removed from num_var") 
	except:
		pass    
	cat_var = df_train.select_dtypes(include=['object','category']).columns
	try:
		cat_var = cat_var.drop(char_var)
		print("char_var removed from cat_var")
	except:
		 pass
	try:
		char_var = char_var.drop(target)
		print("target removed from char_var") 
	except:
		pass
	try:
		cat_var = cat_var.drop(target)
		print("target removed  from cat_var")
	except:
		pass
	try:
		num_var = num_var.drop(target)
		print("target removed  from num_var")
	except:
		pass
	char_var = pd.Index(char_var)
	print('Variables : '); print(variables.values)
	print('Numerical Features : ' ); print(num_var.values)
	print('Categorical Features : ' ); print(cat_var.values)
	print('Name Features : ' ); print(char_var.values)

	# Remove Name/Char Features and other unnecessary Features From DataSet Before EDA
	df_train=df_train.drop(char_var,axis=1)
	if test_file == True:
		df_test=df_test.drop(char_var,axis=1)

	return df_train,df_test,num_var,cat_var,char_var

def EDA(df_train,num_var,cat_var,char_var,target ,target_type='continuos'):


	if target_type == 'continuos':
		# Histogram of Target
		plt.figure(figsize=(10,6))
		sns.distplot(df_train[target], color='g', hist_kws={'alpha': 0.4}, fit=norm)
		plt.title('Histogram of %s' % target)
		plt.show(block=False)
		plt.pause(3)
		plt.close()

	if target_type == 'discrete' :
		pass

	# Numerical Features
	
	if len(num_var)>0:
		## Histograms of Numerical Features with a Normal fit plot added
		f = pd.melt(df_train, value_vars=num_var)
		g = sns.FacetGrid(f, col="variable",  col_wrap=6, sharex=False, sharey=False, height=5)
		g = g.map(sns.distplot, "value" , fit=norm ,color='b',  kde_kws={'bw':0.1},hist_kws={'alpha': 0.4})
		plt.show(block=False)
		plt.pause(7)
		plt.close()

		## Scatterplot of Numerical Features against Target
		f = pd.melt(df_train, id_vars=[target], value_vars=num_var) 
		g = sns.FacetGrid(f, col="variable",  col_wrap=6, sharex=False, sharey=False, height=5)
		g = g.map(sns.regplot, "value", target,color='g')
		plt.show(block=False)
		plt.pause(7)
		plt.close()

	# Categorical Features

	if len(cat_var)>0:
		## Countplots of Categorical Features (Use hue = target , if target is not continuous)
		def countplot(x, **kwargs):
			sns.countplot(x=x)
			x=plt.xticks(rotation=90)
		f = pd.melt(df_train, value_vars=cat_var)
		g = sns.FacetGrid(f, col='variable',col_wrap=6, sharex=False, sharey=False, height=5) # hue = target
		g = g.map(countplot, 'value' )
		plt.show(block=False)
		plt.pause(7)
		plt.close()

		## Box-whisker Plots of Categorical Features against Target  (Possible only if target is continuous)
		def boxplot(x, y, **kwargs):
			sns.boxplot(x=x, y=y)
			x=plt.xticks(rotation=90)
		f = pd.melt(df_train, id_vars=[target], value_vars=cat_var)
		g = sns.FacetGrid(f, col='variable',  col_wrap=6, sharex=False, sharey=False, height=5)
		g = g.map(boxplot, 'value', target)
		plt.show(block=False)
		plt.pause(7)
		plt.close()

		# # Combine Violin Plot & Swarm Plot
		# def swarmviolin(x, y, **kwargs):
		# 	sns.violinplot(x=x, y=y)
		# 	sns.swarmplot(x=x, y=y, color = 'k', alpha = 0.6)
		# 	x=plt.xticks(rotation=90)
		# f = pd.melt(df_train, id_vars=[target], value_vars=cat_var)
		# g = sns.FacetGrid(f, col='variable',  col_wrap=6, sharex=False, sharey=False, height=5)
		# g = g.map(swarmviolin, 'value', target)
		# plt.show(block=False)
		# plt.pause(7)
		# plt.close()

	# Correlation Matrix
	corr = df_train.corr()
	mask = np.zeros_like(corr, dtype=np.bool) # Generate a mask for the upper triangle
	mask[np.triu_indices_from(mask)] = True
	f, ax = plt.subplots(figsize=(11, 9)) # Set up the matplotlib figure
	cmap = sns.diverging_palette(220, 10, as_cmap=True) # Generate a custom diverging colormap
	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr, mask=mask, square=True, linewidths=.5, annot=False, cmap=cmap) 
	plt.yticks(rotation=0)
	plt.title('Correlation Matrix of all Numerical Variables')
	plt.show(block=False)
	plt.pause(7)
	plt.close()
	
	print('Features with absolute correlation values greater than 0.5 are :')
	print(list(corr[(corr >= 0.5) | (corr <= -0.5)].index))
	ax = plt.subplots(figsize=(11, 9))
	sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4) ], cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
				annot=True, annot_kws={"size": 6}, square=False,cbar=True );

	# Correlation with respect to target
	df_corr = pd.DataFrame(corr.nlargest(corr.shape[1],target)[target])
	df_corr = df_corr[(df_corr>=0.5)|(df_corr<=-0.5)].dropna(how = 'all')

	Golden_Features = list(df_corr.index)
	print('Features with absolute correlation values greater than 0.5 wrt Target are : ')
	print(Golden_Features)

	# # PairPlots  
	# sns.set()
	# sns.pairplot(df_train)
	# plt.show(block=False)
	# plt.pause(7)
	# plt.close()

	return Golden_Features, corr

def plot_out_liers(df,cur_var,target):

	plt.scatter(df[cur_var],df[target])
	plt.show(block=False)
	plt.pause(5)
	plt.close()

	scaler = MinMaxScaler(feature_range=(0, 1))
	df[[cur_var,target]] = scaler.fit_transform(df[[cur_var,target]])

	X1 = df[cur_var].values.reshape(-1,1)
	X2 = df[target].values.reshape(-1,1)

	X = np.concatenate((X1,X2),axis=1)
	random_state = np.random.RandomState(42)
	outliers_fraction = 0.05
	# Define seven outlier  tools detectionto be compared
	classifiers = {
			'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
			'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
			'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
			'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
			'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
			'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
			'Average KNN': KNN(method='mean',contamination=outliers_fraction)
	}

	xx , yy = np.meshgrid(np.linspace(0,1 , 200), np.linspace(0, 1, 200))

	for i, (clf_name, clf) in enumerate(classifiers.items()):
		clf.fit(X)
		# predict raw anomaly score
		scores_pred = clf.decision_function(X) * -1
			
		# prediction of a datapoint category outlier or inlier
		y_pred = clf.predict(X)
		n_inliers = len(y_pred) - np.count_nonzero(y_pred)
		n_outliers = np.count_nonzero(y_pred == 1)
		plt.figure(figsize=(10, 10))
		
		# copy of dataframe
		dfx = df
		dfx['outlier'] = y_pred.tolist()
		
		# IX1 - inlier feature 1,  IX2 - inlier feature 2
		IX1 =  np.array(dfx[cur_var][dfx['outlier'] == 0]).reshape(-1,1)
		IX2 =  np.array(dfx[target][dfx['outlier'] == 0]).reshape(-1,1)
		
		# OX1 - outlier feature 1, OX2 - outlier feature 2
		OX1 =  dfx[cur_var][dfx['outlier'] == 1].values.reshape(-1,1)
		OX2 =  dfx[target][dfx['outlier'] == 1].values.reshape(-1,1)
			 
		print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)
			
		# threshold value to consider a datapoint inlier or outlier
		threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
			
		# decision function calculates the raw anomaly score for every point
		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
		Z = Z.reshape(xx.shape)
			  
		# fill blue map colormap from minimum anomaly score to threshold value
		plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
			
		# draw red contour line where anomaly score is equal to thresold
		a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
			
		# fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
		plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
			
		b = plt.scatter(IX1,IX2, c='white',s=20, edgecolor='k')
		
		c = plt.scatter(OX1,OX2, c='black',s=20, edgecolor='k')
		   
		plt.axis('tight')  
		
		# loc=2 is used for the top left corner 
		plt.legend(
			[a.collections[0], b,c],
			['learned decision function', 'inliers','outliers'],
			prop=matplotlib.font_manager.FontProperties(size=20),
			loc=2)
		  
		plt.xlim((0, 1))
		plt.ylim((0, 1))
		plt.title(clf_name)
		plt.show(block=False)
		plt.pause(5)
		plt.close()

def out_lier_score(df,target,num_var):

	scaler = MinMaxScaler(feature_range=(0, 1))
	df = scaler.fit_transform(df.loc[:,num_var],df[target])#.to_numpy()
	random_state = np.random.RandomState(42)
	outliers_fraction = 0.05

	X = df
	df_out_score = []
	# Define seven outlier  tools detectionto be compared
	classifiers = {
			'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
			'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
			'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
			'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
			'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
			'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
			'Average KNN': KNN(method='mean',contamination=outliers_fraction)
	}
	for i, (clf_name, clf) in enumerate(classifiers.items()):
		clf.fit(X)
		# predict raw anomaly score
		scores_pred = clf.decision_function(X) * -1	
		# prediction of a datapoint category outlier or inlier
		y_pred = clf.predict(X)
		df_out_score.append(y_pred.tolist())
		
	df_out_score = pd.DataFrame(df_out_score).T
	df_out_score.columns = list(classifiers.keys())
	return df_out_score

def run(test_file,val_size,n_jobs,target,cur_on,cur_var,char_var,path_train,Definite_vars_to_remove,selection_params,methods,path_test=None):

	# Import Data
	print('################## Importing Data ##################')
	df_train = pd.read_csv(path_train)
	df_train = df_train.replace(r'^\s+$', np.nan, regex=True) # Replacing empty spaces with Null values
	if len(Definite_vars_to_remove)>0:
		df_train=df_train.drop(Definite_vars_to_remove,axis =1)     
	if test_file == True:		
		df_test = pd.read_csv(path_test)
		# df_test[target] = int(0)
		df_test = df_test.replace(r'^\s+$', np.nan, regex=True) # Replacing empty spaces with Null values
		df_test=df_test.drop(Definite_vars_to_remove,axis =1)     
		# df = pd.concat([df_train, df_test])

	# Glimpse of Data
	print('################## Glimpse Data ##################')
	glimpse(df_train) 
	if test_file == True:
		glimpse(df_test)

	# Cleaning Data 
	print('################## Cleaning Data ##################')
	if test_file == True:
		df_train,df_test,num_var,cat_var,char_var = cleanup(test_file,target,char_var,df_train,df_test)
	else:
		df_train,df_test,num_var,cat_var,char_var = cleanup(test_file,target,char_var,df_train)

	# EDA
	print('################## Exploratory Data Analysis ##################')
	# Golden_Features, corr_matrix = EDA(df_train,num_var,cat_var,char_var,target ,target_type='continuos')

	# Outlier Detection
	print('################## Outlier Detection ##################')
	if cur_on:
		df_send = df_train.copy()
		plot_out_liers(df_send,cur_var,target)
	# df_send = df_train.copy()
	# df_out_score = out_lier_score(df_send,target,num_var)
	# print('No of Outliers : ' + str(np.sum(df_out_score.sum(axis=1)>=3)))
	# df_train = df_train.loc[df_out_score.sum(axis=1)<3,:]

	# Feature Engineering
	print('################## Feature Engineering ##################')
	y = df_train.iloc[:,df_train.columns==target]

	df_send = df_train.copy(deep=True)

	fs = FeatureSelector(data = df_send.iloc[:,df_send.columns!=target], labels = df_send.iloc[:,df_send.columns==target])
	fs.identify_all(selection_params = selection_params)

	methods_to_use = []
	for k,v in methods.items():
		if v:
			methods_to_use.append(k)
	x = fs.remove(methods = methods_to_use, keep_one_hot = False)

	df_train = df_train[x.columns]
	df_train['train'] = 1
		
	if test_file == True:
		df_test = df_test[x.columns]
		df_test['train'] = 0
		df = pd.concat([df_train, df_test])
		df = pd.get_dummies(df,drop_first=True)
		df_train = df[df['train']==1]
		df_test = df[df['train']==0]
		df_test = df_test.drop('train',axis=1)
	if test_file ==False:
		df_train = pd.get_dummies(df_train)

	df_train = df_train.drop('train',axis=1)

	print('Features Used are : ' + str(df_train.columns))

	X = df_train.iloc[:,df_train.columns!=target]
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = val_size, random_state = 10)

	scX = StandardScaler()
	X_train = scX.fit_transform(X_train)
	X_valid = scX.transform(X_valid)
	if test_file == True:
		X_test = scX.transform(df_test)

	if selection_params['task'] == 'classification':
		pass
	elif selection_params['task'] == 'regression':
		scy = StandardScaler()
		y_train = scy.fit_transform(y_train)
		y_valid = scy.transform(y_valid)

	# Building Model
	print('################## Building Model ##################')
	if selection_params['task'] == 'classification':
		clf = xgb.XGBClassifier(
						 colsample_bytree=0.2,
						 gamma=0.0,
						 learning_rate=0.01,
						 max_depth=4,
						 min_child_weight=1.5,
						 n_estimators=7200,                                                                  
						 reg_alpha=0.9,
						 reg_lambda=0.6,
						 subsample=0.2,
						 seed=42,
						 silent=1
						 )
		# Run KFold prediction on training set to get a rough idea of how well it does.
		kfold = KFold(n_splits=5)
		results = cross_val_score(clf, X, y.values.reshape(y.shape[0],), cv=kfold)
		print("XGBoost Accuracy score on Training set: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

		clf.fit(X_train, y_train.to_numpy().ravel())
		y_pred = clf.predict(X_train)
		print("XGBoost rmse_score on Training set: ", rmse(y_train, y_pred))
		y_pred = clf.predict(X_valid)
		print("XGBoost rmse_score on Validation set: ", rmse(y_valid, y_pred))
		if test_file == True:
			y_pred = clf.predict(X_test)
			print("Test set predictions saved in output")

	elif selection_params['task'] == 'regression':
		reg = xgb.XGBRegressor(
						 colsample_bytree=0.2,
						 gamma=0.0,
						 learning_rate=0.01,
						 max_depth=4,
						 min_child_weight=1.5,
						 n_estimators=7200,                                                                  
						 reg_alpha=0.9,
						 reg_lambda=0.6,
						 subsample=0.2,
						 seed=42,
						 silent=1
						 )
		# Run KFold prediction on training set to get a rough idea of how well it does.
		kfold = KFold(n_splits=5)
		results = cross_val_score(reg, X, y, cv=kfold)
		print("XGBoost Accuracy score on Training set: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
		reg.fit(X_train, y_train)
		y_pred = reg.predict(X_train)
		print("XGBoost rmse_score on Training set: ", rmse(y_train, y_pred))
		y_pred = reg.predict(X_valid)
		print("XGBoost rmse_score on Validation set: ", rmse(y_valid, y_pred))
		if test_file == True:
			y_pred = reg.predict(X_test)
			print("Test set predictions saved in output")



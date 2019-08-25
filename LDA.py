def get_top_n_words(tf_matrix,n):
    """
        Get a list of the top n words with the highest frequency from each row in the tf matrix
    """
    words = []
    for row in tf_matrix:
        sortMat = {}
        for ind,val in enumerate(row):
            sortMat[ind] = val
        for i in sorted(sortMat.values())[-n:]:
            index = list(sortMat.keys())[list(sortMat.values()).index(i)]
            words.append(index)
            del sortMat[index]

    words = list(set(words))

    return words


def find_best_LDA(tf_matrix,num_topics):
    """
        Grid searching different parameters of LDA to find best model
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.model_selection import GridSearchCV
    params = {'n_components':num_topics}
    lda = LatentDirichletAllocation(max_doc_update_iter=200,learning_method='batch',max_iter=200,evaluate_every=10,perp_tol=1e-4)
    t0 = time.time()
    model = GridSearchCV(lda,param_grid=params,cv=5)
    model.fit(tf_matrix)
    best_model = model.best_estimator_
    print('time taken = %d',time.time()-t0)
    #print(model.best_params_); print(model.best_score_)
    #np.save('grid_scores.npy',model.cv_results_)

    return best_model


def print_top_n_words_best_LDA(tf_vectorizer,best_lda,n_top_words=15):
    tf_words = tf_vectorizer.get_feature_names()
    for idx, topic in enumerate(best_lda.components_):
        message = "Topic %d: " % (idx)
        message += ", ".join([tf_words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

    return


def nth_dominant_topic_best_LDA(best_lda,best_lda_output,num,n=1):
    topics = ["Topic" + str(i) for i in range(best_lda.n_components)]
    docs = ["Doc" + str(i) for i in range(num)]
    df_doc_topic = pd.DataFrame(best_lda_output,columns=topics,index=docs)
    dominant_topic = np.argsort(df_doc_topic.values,axis=1)
    df_doc_topic['dominant_topic_num'] = dominant_topic[:,-n]

    m = scipy.stats.mode(dominant_topic)
    print(m)
    plt.bar(m[0].flatten(),m[1].flatten())
    plt.show()
    
    return df_doc_topic


def topic_distribution_best_LDA(df_doc_topic):
    topic_distr = df_doc_topic['dominant_topic_num'].value_counts().reset_index(name="Num Docs")
    topic_distr.columns = ['Topic Num','Num Docs']
    unique_topics = len(df_doc_topic['dominant_topic_num'].value_counts().index.tolist())

    return topic_distr, unique_topics


def cluster_docs(best_lda_output,clusters=0,method='kmeans'):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=clusters)
    km.fit(best_lda_output)
    ypred = km.predict(best_lda_output)

    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(n_clusters=clusters).fit_predict(best_lda_output)
    
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=2)
    lda_svd = svd.fit_transform(best_lda_output)
    x = lda_svd[:,0]
    y = lda_svd[:,1]

    print(svd.explained_variance_ratio_)
    plt.figure
    if method == 'kmeans':
        plt.scatter(x,y, c=ypred, s=50, cmap='viridis')
    elif method == 'spectral':
        plt.scatter(x,y, c=sc, s=50, cmap='viridis')
    plt.xlabel('comp 1')
    plt.ylabel('comp 2')
    plt.title('topic clusters')
    plt.show()

    return
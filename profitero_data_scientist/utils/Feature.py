

class Feature:

    def __init__(self):
        pass

    def ddd(self):
        # vect = CountVectorizer().fit(X_train[Field.review_en])
        # vect = CountVectorizer(min_df=3, ngram_range=(3, 4)).fit(X_train[predictor])
        # vect_review = CountVectorizer(ngram_range=(2, 3)).fit(X_train[Field.review_tokenized])
        # vect_product = CountVectorizer(ngram_range=(2, 3)).fit(X_train[Field.product_name_tokenized])
        vect_review = CountVectorizer().fit(X_train[Field.review_tokenized])
        vect_product = CountVectorizer().fit(X_train[Field.product_name_tokenized])

        # vect = TfidfVectorizer(min_df=5).fit(X_train[predictor])

        def sprs_matr_to_df(matr):
            matr_to_df = pd.DataFrame()
            for i in range(matr.shape[0]):
                matr_to_df = matr_to_df.append(pd.DataFrame(matr[i, :].toarray()))
            matr_to_df.reset_index(inplace=True)
            return matr_to_df

        X_train_vectorized = sprs_matr_to_df(vect_review.transform(X_train[Field.review_tokenized]))
        tmp_df = sprs_matr_to_df(vect_product.transform(X_train[Field.product_name_tokenized]))
        X_train_vectorized = X_train_vectorized.merge(tmp_df, left_index=True, right_index=True, how='left')
        X_train_vectorized[Field.created_at_day_of_week] = X_train[Field.created_at_day_of_week].values

        X_test_vectorized = sprs_matr_to_df(vect_review.transform(X_test[Field.review_tokenized]))
        tmp_df = sprs_matr_to_df(vect_product.transform(X_test[Field.product_name_tokenized]))
        X_test_vectorized = X_test_vectorized.merge(tmp_df, left_index=True, right_index=True, how='left')
        X_test_vectorized[Field.created_at_day_of_week] = X_test[Field.created_at_day_of_week].values

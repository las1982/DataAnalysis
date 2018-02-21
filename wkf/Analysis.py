import pandas as pd


class Analysis:

    def __init__(self, data, df):
        self.data = data
        self.df = df
        self.input_cols = list(self.data.input_cols_from_metadata)
        self.output_cols = list(self.data.output_cols_from_metadata)
        # self.skipped_cols = list(self.data.cols_in_datasets_but_not_in_metadata)

    def describe(self, top):
        print 'constructing describe...'
        descr = None
        for col in self.df.columns:
            groups = self.df[col].drop_duplicates()
            group_sizes = []
            for group in groups:
                group_sizes.append(self.df[col][self.df[col] == group].size)
            group_count = pd.DataFrame({'group': groups, 'count': group_sizes}).sort_values(by='count', ascending=False)
            df_row = pd.DataFrame(columns=['column', 'count', 'unique'], index=[1])
            df_row['column'] = col
            df_row['count'] = self.df[col].size
            df_row['unique'] = groups.size
            for i in range(top):
                if i >= group_count['group'].size:
                    df_row['top_' + str(i + 1)] = ''
                    df_row['count_' + str(i + 1)] = ''
                else:
                    df_row['top_' + str(i + 1)] = group_count['group'][i] if group_count['group'][
                                                                                 i] != '' else '_empty_'
                    df_row['count_' + str(i + 1)] = group_count['count'][i]

            if descr is None:
                descr = pd.DataFrame(df_row, columns=df_row.columns)
            else:
                descr = descr.append(df_row, ignore_index=True)
            descr = descr.sort_values(by='unique', ascending=True)  # TODO sorting does not work
        print 'constructed data frame', descr.shape
        return descr

    def make_hyperlink(self, cols):
        colss = []
        for i in range(len(cols)):
            if cols[i] in self.df.columns:
                colss.append('=HYPERLINK("#ch_' + str(self.df.columns.get_loc(cols[i])) + '", "' + cols[i] + '")')
            else:
                colss.append('!!! ' + cols[i] + ' is not in the data set!!!')
                print '!!! ' + cols[i] + ' is not in the data set!!!'
        return colss

    def browsable_describe(self):
        browsable_describe = self.df.describe(include='all').T
        browsable_describe['link'] = browsable_describe.index
        browsable_describe['link1'] = browsable_describe['link'].apply(lambda x: '=HYPERLINK("#ch_' + str(self.df.columns.get_loc(x)) + '", "' + x + '")')
        browsable_describe['input/output/skip'] = browsable_describe['link'].apply(lambda x: "input" if x in self.input_cols else "output" if x in self.output_cols else "skip")
        browsable_describe['percent'] = browsable_describe['freq'] / browsable_describe['count']
        cols_to_use = ['input/output/skip', 'link1', 'count', 'unique', 'top', 'freq', 'percent']
        browsable_describe.reset_index(drop=True, inplace=True)
        browsable_describe = browsable_describe[cols_to_use]
        return browsable_describe

    def add_stars_column(self, df):
        dff = df
        dff['vis'] = (dff['count'] / dff['count'].sum() * 100)
        dff['vis1'] = dff['vis'].astype(int)
        dff['vis2'] = '*'
        dff['vis3'] = '*' + dff['vis2'] * dff['vis1']
        dff = dff[['count', 'vis']]
        return dff

    def add_percentage_column(self, df):
        dff = df
        dff['vis'] = dff['count'] / dff['count'].sum()
        dff = dff[['count', 'vis']]
        return dff

    def make_old_table_of_content(self):
        input_cols = self.make_hyperlink(self.input_cols)
        output_cols = self.make_hyperlink(self.output_cols)
        skipped_cols = self.make_hyperlink(self.skipped_cols)
        dff = pd.DataFrame({
            'input': pd.Series(input_cols).sort_values(),
            'output': pd.Series(output_cols).sort_values(),
            'skipped': pd.Series(skipped_cols).sort_values()
        })
        dff.index = dff.index + 1
        return dff

    def make_browsable_analysis(self, output_analysis_file):

        print 'constructing analysis file...'
        writer = pd.ExcelWriter(output_analysis_file, engine='xlsxwriter')
        workbook = writer.book
        percent_format = workbook.add_format({'num_format': '#,##0.00%'})
        self.browsable_describe().to_excel(writer, sheet_name='descr')
        writer.sheets['descr'].set_column('H:H', None, percent_format)
# constructing an old table of content:
#         dff = self.make_old_table_of_content()
#         dff.to_excel(writer, sheet_name='cont')

        col_num = 1
        for col_name in self.df.columns:
            dff = pd.DataFrame(self.df[col_name])  # .groupby(col_name).count()
            dff['count'] = 1
            dff = dff.groupby(col_name).count()
            dff = dff.sort_values(by='count', ascending=False)
            dff = self.add_percentage_column(dff)
            dff.columns = ['Total groups: ' + str(dff.index.size),
                           '=HYPERLINK("#' + 'descr' + '", "' + 'Back to content' + '")']
            dff.to_excel(writer, sheet_name='ch_' + str(self.df.columns.get_loc(col_name)))
            writer.sheets['ch_' + str(self.df.columns.get_loc(col_name))].set_column('C:C', None, percent_format)
            col_num = col_num + 1
        writer.save()


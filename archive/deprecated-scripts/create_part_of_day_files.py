from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider


def create_files(patients:list):
    for patient in patients:
        provider = OhioBgcProvider(ohio_no=patient)
        df = provider.tsfresh_dataframe()[['part_of_day', 'bg_value']]
        df['part_of_day'] = df['part_of_day'].replace(
            ['late night', 'night', 'morning', 'afternoon', 'evening'],
            ['5.Late Night', '4.Night', '1.Morning', '2.Afternoon', '3.Evening']
        )
        print(df.head())
        df.to_excel(f'{patient}_part_of_day.xlsx')
        
        
if __name__ == '__main__':
    
    # * Declare here the OHIO dataset patient ids to process
    patients = [559, 563, 570, 575, 588, 591]
    
    create_files(patients)
    
    
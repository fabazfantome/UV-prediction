from datetime import datetime, timedelta
from starstream import SDO, DataDownloading
from dateutil.relativedelta import relativedelta

def datelist_get(in_date: datetime, max_date: datetime, deltat: relativedelta):
    datelist = []
    date1 = in_date
    deltatt = timedelta(days= 1)
    while True:
        if date1 >= max_date - deltat:
            datelist.append((date1, max_date))
            break
        datelist.append((date1, date1 + deltat))
        date1 += deltat + deltatt
    return datelist

datelist = datelist_get(datetime(2024, 1, 1), datetime(2024, 10, 1), timedelta(days= 7))
print(datelist)
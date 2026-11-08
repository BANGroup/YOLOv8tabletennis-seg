import datetime


def input_paths(date, depth):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    return [
        f"/user/s27560962/data/events/date={(dt - datetime.timedelta(days=offset)).strftime('%Y-%m-%d')}/event_type=message"
        for offset in range(depth)
    ]

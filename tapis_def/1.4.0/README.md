# Note for APP Definition and Job Definition

## App Definition

You can use the same app ID with different versions.

## Job Definition

If you need to run your job with a certain reservation, you have to use the exact 'partition' or 'queue' + 'allocation' defined in the reservation.
A sample full reservation information may look like this:

```
reservationname=GHTapis+Nairr  # -> this is the reservation name you have to put after scheduler option `--reservation`
Starttime=2026-03-10T08:15
endtime=2026-03-10T17:45
partition=gh  # -> this is the partition/queue name you have to put for logicalQueue
nodecount=45
accounts=tra24006  # -> This is the allocation you have to put for your `-A` scheduler option.
Reservation created: GHTapis+Nairr
```

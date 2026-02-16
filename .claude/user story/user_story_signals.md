
This is a Comprehensive User Story, how now metrics after Iterative regression are used to generate signals. 

Current Situation.

Now I have Power BI report where Python is inside Power Query and commits Iterative regression.
This Project is mainly replication of Power BI logic. 

Power BI Mental Framework of Signal Generation:


Most Important we are Looking to Angle metric. 

We look whethere angle oscalator is reversing. 
Either it goes from 0 to peak, or from peak to 0. 
Meaning close to 0 and peak positions 
are potential reversals. Manipulation is when we are locally having chaotic reversals.
So we need to confirm the Reversal using acceleration = lag of angle - angle. If using GMM it is distant or very distant class (recall GMM segmentation in this Project) then we have high quality signals. Otherwise low quality and False Positive signals. 

So We know reversal and we know filter signals using GMM class of acceleration.

However to confirm the direction we use forward slope of the corresponding timeframe and window - to 
register the event. We have Strong Signal event and we add direction from Forward slope. So now we have either     Strong Signal Buy or Strong Signal Sell. 

To generate expected percentage of Long or short we use always concept.

Mistique Oracul Concept

During my exploration I have noticed that angles crossing signals work like fortune teller.
Especially where we have Convergence of the same event in multiple time frames.

(Refer to pngs in .cloude folder) use cross in naming search to find the right one.

This Should be explored more. But for now it we have it as hidden weapon against manioulation.
Since if we are cobining all together we can see a real direction.

Notes:

The Strongest Signals Usually when different window size timeframe angles are highly correlated or equal.

Expectations:

Claude should Implement below Tasks.

Analyze current situation in project and make sure whether we do everyhting correctly. if no

- Alighn everything with user story described in this file. Especially strategy of signal processing.
- Calculate predcited next values for angle , forward slope, accleration. use simple formula 

c = current, 
a = previous
b = previous of previous
def child(a, b, c):
    angle = c - b
    curve = abs(a - 2*b + c)
    ratio = min(abs(c) / (abs(b) + 1), 1.5)
    return c + angle * ratio / (1 + curve/2)

After prediction - predict once more using for current the predicted value. 
overall we end up with two step prediction to the future. 

if we have a, b, c, we predict cc,  ccc.

- Based on this prections we should create calendar table in streamlit. 

The prupose of calendar is to predict signal befor it happened based on convergence 

How calendar looks like. 

on rows we have 

timeframe-window

15min - df
30min - df 
....

on columns we have rolling last 3 hours in 15 min candle seperation of live data + 4h step forward of now. 

We use 4h step forward to put there predicted convergences .

as a metric we need to see. Convergents event count, Predicted Signal.


Reversal for Angle only:

def detect_cycle_events(angles:pd.Series, window=5):

    angles = angles.values
    events = np.zeros(len(angles))  # Initialize event array with 0s

    for i in range(len(angles) - window + 1):
        window_vals = angles[i:i + window]

        # Check if strictly decreasing or increasing (Normal cycle phase)
        if all(window_vals[j] > window_vals[j + 1] for j in range(window - 1)) or \
           all(window_vals[j] < window_vals[j + 1] for j in range(window - 1)):
            events[i + window - 1] = 0  # No event
        else:
            # Check for cycle break (bottom or peak formation)
            if window_vals[0] > window_vals[1] > window_vals[2] > window_vals[3] < window_vals[4]:
                events[i + window - 1] = 1  # Bottom detected
            elif window_vals[0] < window_vals[1] < window_vals[2] < window_vals[3] > window_vals[4]:
                events[i + window - 1] = 1  # Peak detected

    return events

---

## Original Signal Logic Notes

current signal generation logic is:

we use angle and forward slope only. + angle acceleration for confirmation and signal filtering

1. Reversal signal - when angle is going from top to botton 0, hills-peaks are reversal signal as well
2. Direction - use forward slope to see whether the reversal is going up or down
3. acceleration, should be ssegmented using gmm to understand very close points, close points, baseline, distant points, very distant points.

Note:

Good signals happen when we have  distant points, very distant points
Reversal happens - peak or 0 point
There is clear direction.

Also We have very strong signals , similar to crossing MA's, but here it is
crossing angles. I did not examined them completely but they work as they really predict future.

for example if higher window size angle df1 hits to df . df1 going down and df up we habe short signal.

if df is hoing down and df1 is going up , when they cross its a long signal.

So we use this concept on all window sizes and on all timeframes.

Note

Higher the Timeframe , higher the impact.

If we have multiple convergences of the same event using this metho this is high quality signal.


CREATE TABLE marts.all_events AS
SELECT id, 'web' AS channel FROM raw.web_events
UNION ALL
SELECT id, 'app' AS channel FROM raw.app_events;

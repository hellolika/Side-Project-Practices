TRUNCATE TABLE [dbo].[Team]
GO

BEGIN
	SET IDENTITY_INSERT team ON;

    INSERT INTO [dbo].[Team] (
        [TeamId],
        [TeamName],
        [StartTime],
        [EndTime],
        [TotalHour],
        [IsEnable],
        [DepartmentId]
    ) 
    VALUES
    (1, 'TW', '08:30', '17:30', 8.000000000, 1, 1),
    (6, 'TW-Finance', '08:30', '17:30', 8.000000000, 1, 1);

   SET IDENTITY_INSERT team OFF;

END
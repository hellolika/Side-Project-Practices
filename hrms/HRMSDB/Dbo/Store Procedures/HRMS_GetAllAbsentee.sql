CREATE PROCEDURE [dbo].[HRMS_GetAllAbsentee.2.0.0]
	@tvpWorkingDay [dbo].[WorkingDay] Readonly
AS
BEGIN
	
	DECLARE @AbsenteeTemp TABLE ([Id] int, [Username] NVARCHAR(50), [TeamId] INT, 
	[WorkDate] DATE, [LeaveId] INT DEFAULT 0, [LeaveType] NVARCHAR(50) DEFAULT 'Absent',
	[ClockInTime] DATE NULL, [ClockOutTime] DATE NULL)
	
	INSERT INTO @AbsenteeTemp ([Id],[Username],[TeamId],[WorkDate])
	SELECT m.[Id] , m.[Username] , m.[TeamId], t.[WorkDate]
	FROM [dbo].[Member] m WITH(NOLOCK) CROSS JOIN @tvpWorkingDay t 
	WHERE  NOT EXISTS
	(SELECT TOP 1 1
	FROM [dbo].[Attendances] a WITH(NOLOCK)
	WHERE a.[MemberId] = m.[Id] AND a.[WorkDate] = t.[WorkDate]) 
	and  NOT EXISTS
	(SELECT TOP 1 1
	FROM [dbo].ReClockRecord r WITH(NOLOCK)
	WHERE r.[MemberId] = m.[Id] AND r.Date = t.[WorkDate])
	and m.Permission = 0

	IF EXISTS(SELECT TOP 1 1 FROM [dbo].[ReClockRecord] r WITH(NOLOCK) JOIN @AbsenteeTemp a ON r.[MemberId] = a.[Id] WHERE r.[Date] = a.[WorkDate]
	AND r.[Status] = 1 AND r.[MemberId] = a.[Id])
	BEGIN
		DELETE FROM @AbsenteeTemp  
		WHERE [WorkDate] IN (SELECT [Date] FROM [dbo].[ReClockRecord] r WITH(NOLOCK) WHERE r.[MemberId] = [Id] AND r.[Status] = 1 AND  r.[Date] = [WorkDate])
	END

	IF EXISTS(SELECT TOP 1 1 FROM [dbo].[TakeLeaveRecord] l WITH(NOLOCK) 
		INNER JOIN @AbsenteeTemp a ON l.[MemberId] = a.[Id] WHERE CONVERT(Date, l.[ENDDATE]) >= a.[WorkDate]
		AND CONVERT(Date, l.[StartDate]) <= a.[WorkDate] AND l.[Status] = 1)
	BEGIN
		UPDATE @AbsenteeTemp 
		SET [LeaveId] = t.[LeaveType],
			[LeaveType] = lt.[Type]
		FROM [dbo].[TakeLeaveRecord] t WITH(NOLOCK)
		INNER JOIN [dbo].[LeaveType] lt WITH(NOLOCK)
		ON lt.[TypeId] = t.[LeaveType]
		WHERE t.[Status] = 1 AND CONVERT(Date,t.[StartDate]) <= [WorkDate] 
		AND CONVERT(Date, t.[EndDate]) >= [WorkDate] AND t.[MemberId] = [Id]
	END

	SELECT [Id], [Username], a.[TeamId], t.[TeamName], 
	[WorkDate], [LeaveId], [LeaveType] 
	FROM @AbsenteeTemp a INNER JOIN [dbo].[Team] t WITH(NOLOCK) 
	ON t.[TeamId] = a.[TeamId]
END
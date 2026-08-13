CREATE PROCEDURE [dbo].[HRMS_GetAllAttendance.1.0.0]
	@date DATE
AS
BEGIN
	SELECT m.[Id] as MemberId,
	m.[Username],
	t.[TeamName],
    (CASE WHEN att.[WorkDate] IS NULL THEN rcrDate.[Date] ELSE att.[WorkDate] END) AS WorkDate,
	att.[ClockIn],
	att.[ClockOut],
	att.[ClockInLocation],
	att.[ClockOutLocation],
	att.[ClockInRemark],
	att.[ClockOutRemark],
    rcrIn.[Time] AS ReClockIn,
    rcrOut.[Time] AS ReClockOut,
	m.Email,
	su.Id as SlackId,
	su.Username as SlackUsername,
	su.Realname as SlackRealName
	FROM [dbo].[Attendances] att
	RIGHT JOIN [dbo].[Member] m ON m.[Id] = att.[MemberId] AND m.[IsDeleted] = 0 AND att.[WorkDate] = @date
    LEFT JOIN [dbo].[ReClockRecord] rcrDate on rcrDate.[MemberId] = m.[Id] AND rcrDate.[Date] = @date
    LEFT JOIN [dbo].[ReClockRecord] rcrIn on rcrIn.[MemberId] = m.[Id] AND rcrIn.[Date] = @date  AND rcrIn.[IsClockIn] = 1
    LEFT JOIN [dbo].[ReClockRecord] rcrOut on rcrOut.[MemberId] = m.[Id] AND rcrOut.[Date] = @date AND rcrOut.[IsClockIn] = 0
    LEFT JOIN [SlackUserInfo] su ON su.Email = m.Email
	INNER JOIN [dbo].[Team] t ON m.[TeamId] = t.[TeamId] WHERE m.[IsDeleted] = 0;
END
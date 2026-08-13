CREATE PROCEDURE [dbo].[HRMS_GetAttendanceById.1.0.0]
	@memberId INT,
	@startDate DATE,
	@endDate Date
AS
BEGIN
	SELECT att.[WorkDate] AS [Date], att.[MemberId],
	m.[Username], att.[WorkDate], t.[TeamName],
	att.[ClockIn], att.[ClockOut],
	att.[ClockInLocation], att.[ClockOutLocation],
	att.[ClockInRemark] as ClockInRemark, att.[ClockOutRemark] as ClockOutRemark
	FROM [dbo].[Attendances] att 
	RIGHT JOIN [dbo].[Member] m ON att.[MemberId] = m.[Id]
	LEFT JOIN [dbo].[Team] t ON m.[TeamId] = t.[TeamId]
	WHERE att.[MemberId] = @memberId AND att.[WorkDate] BETWEEN @startDate AND @endDate
END


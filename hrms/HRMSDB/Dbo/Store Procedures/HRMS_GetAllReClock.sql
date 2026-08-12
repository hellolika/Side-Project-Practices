CREATE PROCEDURE [dbo].[HRMS_GetAllReClock.2.0.1]
	@startDate DATE,
	@endDate DATE
AS
BEGIN
	SET NOCOUNT ON;

	SELECT [RequestId], [MemberId], m.[Username], m.[TeamId], t.[TeamName],
	[Date], [Time], [IsClockIn], [Status], [Reason], ub.[Username] AS [UpdateBy]
	FROM [dbo].[ReClockRecord] r WITH(NOLOCK)
	INNER JOIN [dbo].[Member] m WITH(NOLOCK) ON r.[MemberId] = m.[Id]
	LEFT JOIN [dbo].[Team] t WITH(NOLOCK) ON t.[TeamId] = m.[TeamId]
	LEFT JOIN [dbo].[Member] ub WITH(NOLOCK) ON r.[UpdateBy] = ub.[Id]
	WHERE [Date] BETWEEN @startDate AND @endDate
		
	SELECT 0 AS ErrorCode;

END
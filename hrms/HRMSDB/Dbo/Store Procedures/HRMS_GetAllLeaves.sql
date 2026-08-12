CREATE PROCEDURE [dbo].[HRMS_GetAllLeaves.4.0.0]
	@startDate DATE,
	@endDate DATE
AS
BEGIN
	SET NOCOUNT ON;
	
	SELECT tl.[RequestId], tl.[MemberId], m.[Username], m.[TeamId], t.[TeamName],
	tl.[LeaveType] AS [LeaveId], lt.[Type] AS [LeaveType],tl.[Image], tl.[StartDate], tl.[IsCancel],
	tl.[EndDate], tl.[Status], tl.[Reason], tl.[ResponseReason], tl.[SubmittedOn], ub.[Username] AS [UpdateBy],
	tl.[NumberOfDay]
	FROM [dbo].[TakeLeaveRecord] tl WITH(NOLOCK) 
	INNER JOIN [dbo].[Member] m WITH(NOLOCK) ON tl.[MemberId] = m.[Id]
	INNER JOIN [dbo].[LeaveType] lt ON lt.[TypeId] = tl.[LeaveType]
	LEFT JOIN [dbo].[Team] t WITH(NOLOCK) ON m.[TeamId] = t.[TeamId]
	LEFT JOIN [dbo].[Member] ub WITH(NOLOCK) ON tl.[UpdateBy] = ub.[Id]
	WHERE ((tl.[SubmittedOn] BETWEEN @startDate AND @endDate) OR ((tl.[StartDate] BETWEEN @startDate AND @endDate) OR (tl.[EndDate] BETWEEN @startDate AND @endDate)))
	SELECT 0;
END
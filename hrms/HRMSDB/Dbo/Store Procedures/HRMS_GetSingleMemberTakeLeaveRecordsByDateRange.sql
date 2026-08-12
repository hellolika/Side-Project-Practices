CREATE PROCEDURE [dbo].[HRMS_GetSingleMemberTakeLeaveRecordsByDateRange.3.0.1]
	@memberId INT,
	@startDate DATETIME,
	@endDate DATETIME
AS
BEGIN 
	SET NOCOUNT ON;
	SELECT lr.[StartDate], lr.[EndDate], lr.[NumberOfDay], lr.[LeaveType] AS [LeaveId],
	lr.[Status], lt.[Type] AS [LeaveType], lr.[IsCancel], lr.[NumberOfDay]
	FROM [dbo].[TakeLeaveRecord] lr WITH(NOLOCK) 
	LEFT JOIN [dbo].[LeaveType] lt WITH(NOLOCK)
	ON lr.[LeaveType] = lt.[TypeId]
	WHERE lr.[MemberId] = @memberId AND 
	((CAST([StartDate] AS DATE) BETWEEN @startDate AND @endDate) 
	OR (CAST([EndDate] AS DATE) BETWEEN @startDate AND @endDate) 
	OR (@startDate BETWEEN CAST([StartDate] AS DATE) AND CAST([EndDate] AS DATE)) 
	OR (@endDate BETWEEN CAST([StartDate] AS DATE) AND CAST([EndDate] AS DATE)))
END
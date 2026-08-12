CREATE PROCEDURE [dbo].[HRMS_GetLeaveRequestsByMemberId.1.0.0]
	@memberId INT,
	@startDate DATE,
	@endDate DATE
AS
BEGIN
	SELECT * 
	FROM [dbo].[TakeLeaveRecord]
	WHERE [MemberId] = @memberId 
	AND ([StartDate] BETWEEN @startDate AND @endDate OR [EndDate] BETWEEN @startDate AND @endDate)
END
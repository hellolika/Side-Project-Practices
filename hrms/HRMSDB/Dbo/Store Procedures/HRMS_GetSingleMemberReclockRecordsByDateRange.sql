CREATE PROCEDURE [dbo].[HRMS_GetSingleMemberReclockRecordsByDateRange.1.0.0]
	@memberId INT,
	@startDate DATE,
	@endDate DATE
AS
BEGIN 
	SET NOCOUNT ON;
	SELECT [Date], [Time], [IsClockIn], [Status] FROM [dbo].[ReClockRecord] WITH(NOLOCK)
	WHERE [MemberId] = @memberId AND [Date] BETWEEN @startDate AND @endDate
END
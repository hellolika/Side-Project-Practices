CREATE PROCEDURE [dbo].[HRMS_GetSingleMemberAttendancesByDateRange.1.0.0]
	@memberId INT,
	@startDate DATETIME,
	@endDate DATETIME
AS
BEGIN 
	SET NOCOUNT ON;
	SELECT [WorkDate], [ClockIn], [ClockOut] FROM [dbo].[Attendances] WITH(NOLOCK)
	WHERE [MemberId] = @memberId AND [WorkDate] >= @startDate AND [WorkDate] <= @endDate
END
CREATE PROCEDURE [dbo].[HRMS_GetAllReClockRecordsById.1.0.0]
	@memberId INT,
    @startDate DATE,
	@endDate Date
AS
BEGIN
	SET NOCOUNT ON;

	SELECT rcr.[MemberId],m.[Username],t.[TeamName], rcr.[Date], rcr.[Time], rcr.[IsClockIn],rcr.[Location], rcr.[Status], rcr.[Reason] FROM [dbo].[ReClockRecord] rcr WITH(NOLOCK)
	JOIN [dbo].[Member] m ON m.[Id] = @memberId
	JOIN [dbo].[Team] t ON t.[TeamId] = m.[TeamId]
	WHERE [MemberId] = @memberId AND rcr.[Date] BETWEEN @startDate AND @endDate

END